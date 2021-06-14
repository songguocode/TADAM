import os
import sys
sys.path.append(os.getcwd())
import cv2
import numpy as np
import torch

from ..dataset.mot import MOTTracking, collate_fn, read_mot_file
from ..configs.config import load_config
from ..utils.log import log_or_print, get_logger


def voc_color_code(num_colors=100):
    def to_binary(val, idx):
        return ((val & (1 << idx)) != 0)

    color_code = np.zeros((num_colors, 3), dtype=np.uint8)
    for i in range(num_colors):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (to_binary(c, 0) << 7 - j)
            g |= (to_binary(c, 1) << 7 - j)
            b |= (to_binary(c, 2) << 7 - j)
            c >>= 3
        color_code[i, :] = [r, g, b]
    return color_code


def plot_boxes(
    frame,
    boxes,
    obj_ids=None,
    show_ids=True,
    scores=None,
    show_scores=False,
    polygons=None,
    show_polygons=False,
    masks=None,
    show_masks=False,
    show_info=False,
    frame_id=0,
    image_scale=1,
    text_scale=2,
    line_thickness=2,
    fps=0.
):
    """
        Draw a frame with bounding boxes
    """
    im = np.copy(frame)
    im_h, im_w = im.shape[:2]

    # Determin colors first:
    colors = []
    for i in range(len(boxes)):
        if obj_ids is None:
            colors.append((255, 0, 0))
        else:
            obj_id = obj_ids[i]
            # color = get_color(abs(int(obj_id)))
            colors.append(tuple([int(c) for c in voc_color_code(256)[abs(int(obj_id)) % 256]]))

    # Draw masks first
    # Input should be K x H x W of float, where K is number of objects
    if masks is not None and show_masks:
        final_mask = np.zeros_like(im, dtype=np.uint8)
        for i, mask in enumerate(masks):
            mask = np.expand_dims(masks[i], axis=-1)
            final_mask += np.uint8(np.concatenate((mask * colors[i][0], mask * colors[i][1], mask * colors[i][2]), axis=-1))
        im = cv2.addWeighted(im, 0.77, final_mask, 0.5, -1)

    for i, box in enumerate(boxes):
        if obj_ids is not None:
            obj_id = obj_ids[i]
        else:
            obj_id = None

        x1, y1, x2, y2 = box
        intbox = tuple(map(int, (x1, y1, x2, y2)))
        # Draw box
        if not show_polygons:
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=colors[i], thickness=line_thickness)

        # Draw Polygons
        polygon = None
        if polygons is not None and show_polygons:
            polygon = polygons[i]
            if polygon is not None:
                cv2.polylines(im, [polygon.reshape((-1, 1, 2))], True, (0, 255, 0), 3)

        # Draw id at top-left corner of box
        if obj_id is not None and show_ids:
            cv2.putText(im, f"{obj_id:d}", (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_PLAIN,
                text_scale * 0.6, (0, 255, 255), thickness=1)

        # Draw scores at bottom-left corner of box
        score = None
        if scores is not None and show_scores:
            score = scores[i]
            cv2.putText(im, f"{score:.4f}", (int(x1), int(y2) - 20), cv2.FONT_HERSHEY_PLAIN,
                text_scale * 0.6, (0, 255, 255), thickness=1)
    if show_info:
        cv2.putText(im, "frame: %d fps: %.2f num: %d" % (frame_id, fps, len(boxes)), (0, int(15 * text_scale)),
            cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    # Resize
    im = cv2.resize(im, (int(im_w * image_scale), int(im_h * image_scale)))

    return im


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


class Content(object):
    def __init__(self):
        self.list = ["GT", "DET", "RESULT"]
        self.current = self.list[0]

    def next(self):
        self.current = self.list[(self.list.index(self.current) + 1) % len(self.list)]


def show_mot(
    dataloader,
    dataset_root="../datasets",
    dataset="MOT17",
    which_set="train",
    sequence="MOT17-FRCNN-02",
    vis_threshold=0.1,
    result_root="output/results",
    result=None,
    start_frame=1,
    scale=1.0,
):
    """
        Visualize MOT detections/ground truths/tracking results
    """
    content = Content()
    hide_info = False
    hide_ids = False
    save_image = False
    save_dir = "./output/images"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Load data
    include_result = False
    if result is not None:
        include_result = True
        result_path = os.path.join(result_root, dataset, result, f"{sequence}.txt")
        assert os.path.isfile(result_path), f"No valid result file found at '{result_path}'"
        log_or_print(logger, f"Loaded result file at '{result_path}'")

    # Load all data at once and store
    det_by_frame = []
    gt_by_frame = []
    result_by_frame = []
    for frame_id, batch in enumerate(dataloader):
        frame_id += 1  # Start from 1
        # Get MOT data
        _, det_boxes, det_scores, gt_boxes, gt_ids, gt_visibilities = batch
        # Make a detached copy to stop dataloader from using file
        det_boxes = det_boxes[0].detach().clone().cpu().numpy()
        det_scores = det_scores[0].detach().clone().cpu().numpy()
        gt_boxes = gt_boxes[0].detach().clone().cpu().numpy()
        gt_ids = gt_ids[0].detach().cpu().clone().numpy()
        gt_visibilities = gt_visibilities[0].detach().clone().cpu().numpy()
        del batch

        # Detections
        det_by_frame.append((det_boxes, det_scores))

        # GT
        if gt_boxes is not None:  # In case of test sets
            gt_boxes = [gt_box for i, gt_box in enumerate(gt_boxes) if gt_visibilities[i] > vis_threshold]
            gt_ids = [gt_id for i, gt_id in enumerate(gt_ids) if gt_visibilities[i] > vis_threshold]
            gt_visibilities = [gt_visibility for i, gt_visibility in enumerate(gt_visibilities) if gt_visibilities[i] > vis_threshold]
            gt_by_frame.append((gt_boxes, gt_ids, gt_visibilities))
        else:
            gt_by_frame.append(([], [], []))

        # Result
        if include_result:
            result_boxes, result_ids, result_scores, _ = read_mot_file(result_path, frame_id)
            # In case no result for the frame, empty lists are returned in that case
            if len(result_boxes) and len(result_ids) and len(result_scores):
                result_boxes = result_boxes.cpu().numpy()
                result_ids = result_ids.cpu().numpy()
                result_scores = result_scores.cpu().numpy()
            result_by_frame.append((result_boxes, result_scores, result_ids))
        else:
            result_by_frame.append(([], [], []))

    # Load images
    image_dir = os.path.join(dataset_root, dataset, which_set, sequence, "img1")
    file_list = os.listdir(image_dir)

    def get_index(x_str):
        return x_str[:-4]
    file_list = sorted(file_list, key=get_index)

    # Show
    window_name = f"MOT Visualization - {sequence}"
    frame_id = start_frame
    filename = file_list[frame_id - 1]
    im = draw_frame(image_dir, filename, frame_id, det_by_frame[frame_id - 1],
        gt_by_frame[frame_id - 1], result_by_frame[frame_id - 1], content.current, hide_info, hide_ids, scale)
    while True:
        cv2.imshow(window_name, im)
        # Save if toggled
        if save_image:
            cv2.imwrite(os.path.join(save_dir, f"{sequence}-{content.current}-{filename.split('.')[0]}.jpg"), im)
        key = cv2.waitKey(0)
        # Prev frame, press Key "<"
        if key == 44:
            frame_id = (frame_id - 1) % len(file_list)
            if frame_id == 0:
                frame_id = len(file_list)
            filename = file_list[frame_id - 1]
            im = draw_frame(image_dir, filename, frame_id, det_by_frame[frame_id - 1],
                gt_by_frame[frame_id - 1], result_by_frame[frame_id - 1], content.current, hide_info, hide_ids, scale)
        # Next frame, press Key ">"
        elif key == 46:
            frame_id = (frame_id + 1) % len(file_list)
            if frame_id == 0:
                frame_id = len(file_list)
            filename = file_list[frame_id - 1]
            im = draw_frame(image_dir, filename, frame_id, det_by_frame[frame_id - 1],
                gt_by_frame[frame_id - 1], result_by_frame[frame_id - 1], content.current, hide_info, hide_ids, scale)
        # Exit, press Key "q" or Esc
        elif key == 113 or key == 27:
            break
        # Other options
        else:
            # Rotate among GT, DET, RESULT, press Key "t"
            if key == 116:
                content.next()
            # Save crops, press Key "c"
            elif key == 99:
                if content.current == "GT":
                    boxes_info = gt_by_frame[frame_id - 1]
                elif content.current == "DET":
                    boxes_info = det_by_frame[frame_id - 1]
                elif content.current == "RESULT":
                    boxes_info = result_by_frame[frame_id - 1]
                save_crops(image_dir, filename, frame_id, boxes_info, content.current, save_dir, sequence)
            # Save image, press key "s"
            elif key == 115:
                save_image = not save_image
            # Hide info in image, press key "h"
            elif key == 104:
                hide_info = not hide_info
            # Hide ids in image, press key "i"
            elif key == 105:
                hide_ids = not hide_ids
            im = draw_frame(image_dir, filename, frame_id, det_by_frame[frame_id - 1],
                gt_by_frame[frame_id - 1], result_by_frame[frame_id - 1], content.current, hide_info, hide_ids, scale)


def draw_frame(
    image_dir,
    filename,
    frame_id,
    detections,
    groundtruths,
    results,
    content_selection,
    hide_info,
    hide_ids,
    scale
):
    """
        Draw a frame with given detections/ground truths/results
    """
    im = cv2.imread(os.path.join(image_dir, filename))
    content_info_position = (0, 22)
    content_info_color = (0, 255, 255)
    content_info_thickness = 2

    if content_selection == "DET":
        if not hide_info:
            cv2.putText(im, f"Frame: {frame_id:5d}   Detections", content_info_position,
                cv2.FONT_HERSHEY_PLAIN, content_info_thickness, content_info_color, thickness=2)
        im = plot_boxes(im, detections[0], scores=detections[1], show_scores=True, image_scale=scale)
    elif content_selection == "GT":
        if not hide_info:
            cv2.putText(im, f"Frame: {frame_id:5d}   Grount Truths", content_info_position,
                cv2.FONT_HERSHEY_PLAIN, content_info_thickness, content_info_color, thickness=2)
        im = plot_boxes(im, groundtruths[0], obj_ids=groundtruths[1], show_ids=not hide_ids,
            scores=groundtruths[2], show_scores=True, image_scale=scale)
    elif content_selection == "RESULT":
        if not hide_info:
            cv2.putText(im, f"Frame: {frame_id:5d}   Results", content_info_position,
                cv2.FONT_HERSHEY_PLAIN, content_info_thickness, content_info_color, thickness=2)
        im = plot_boxes(im, results[0], scores=results[1], obj_ids=results[2], show_ids=not hide_ids, image_scale=scale)
    return im


def save_crops(image_dir, filename, frame_id, boxes_info, content, save_dir, sequence):
    im = cv2.imread(os.path.join(image_dir, filename))
    height, width = im.shape[:2]
    try:
        for i, box in enumerate(boxes_info[0]):
            log_or_print(logger, f"ID {boxes_info[1][i]} box: {boxes_info[0][i]}")
            crop = im[max(0, int(box[1])):min(height, int(box[1] + box[3])), max(0, int(box[0])):min(width, int(box[0] + box[2])), :]
            cv2.imwrite(os.path.join(save_dir, f"{sequence}-{content}-FRAME-{frame_id}-ID-{boxes_info[1][i]}.jpg"), crop)
    except Exception:
        log_or_print(logger, "No bounding boxes in current content", level="warning")


if __name__ == "__main__":
    """
        Run a visualization demo with parameters
    """
    # arguments
    import argparse
    parser = argparse.ArgumentParser(description="MOT Visualization")
    parser.add_argument("--config", default="TADAM_MOT17", type=str, help="config file to load")
    parser.add_argument("--which-set", default="train", type=str, choices=["train", "test"], help="which sequence")
    parser.add_argument("--sequence", default="02", type=str, help="which sequence")
    parser.add_argument("--public-detection", default="FRCNN", type=str,
        choices=["None", "DPM", "FRCNN", "SDP"], help="public detection")
    parser.add_argument("--result", default=None, type=str, help="name for loading results")
    parser.add_argument("--start-frame", default=1, type=int, help="start frame")
    parser.add_argument("--scale", default=1, type=float, help="visual size of image")
    parser.add_argument("--vis_threshold", default=0.1, type=float, help="visibility threshold for gt")
    args = parser.parse_args()

    config, cfg_msg = load_config(args.config)
    logger = get_logger(name="global", save_file=False, console_verbose=False)
    log_or_print(logger, cfg_msg)

    public_detection = args.public_detection if config.NAMES.DATASET == "MOT17" else "None"
    dataloader = torch.utils.data.DataLoader(MOTTracking(config.PATHS.DATASET_ROOT,
        config.NAMES.DATASET, args.which_set, args.sequence, public_detection, args.vis_threshold),
        batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    public_detection = f"-{public_detection}" if public_detection != "None" else ""
    sequence = f"{config.NAMES.DATASET}-{int(args.sequence):02d}{public_detection}"

    # Show info
    log_or_print(logger, f"Showing {config.NAMES.DATASET}/{args.which_set}/{sequence}")

    show_mot(dataloader, config.PATHS.DATASET_ROOT, config.NAMES.DATASET, args.which_set, sequence,
        args.vis_threshold, config.PATHS.RESULT_ROOT, args.result, args.start_frame, args.scale)
