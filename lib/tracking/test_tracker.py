import os
import cv2
import torch
import itertools
from ..configs.config import load_config
from .tracker import OnlineTracker
from ..dataset.mot import MOTTracking, collate_fn, get_seq_names
from ..utils.image_processing import tensor_to_cv2
from ..utils.visualization import plot_boxes
from ..utils.log import get_logger, log_or_print
from ..utils.timer import Timer
from ..utils.official_benchmark import benchmark


# Follows MOTChallenge data format, ltwh: left, top, width, height
def write_results(filename, results, logger):
    with open(filename, "w") as f:
        for frame_id, ltwhs, track_ids in results:
            for ltwh, track_id in zip(ltwhs, track_ids):
                if track_id < 0:
                    continue
                l, t, w, h = ltwh
                line = f"{frame_id},{track_id},{l},{t},{w},{h},1,-1,-1,-1\n"
                f.write(line)
    log_or_print(logger, f"Saved results to '{filename}'")


def test_sequence(
    dataloader,
    config,
    logger,
    seq_result_file,
    plot_frames=False,
):
    tracker = OnlineTracker(config, logger)

    timer = Timer()
    results = []
    log_or_print(logger, f"Started processing frames")
    for frame_id, batch in enumerate(dataloader):
        # Start from 1
        frame_id += 1

        # Get detections
        frame, det_boxes, _, _, _, _ = batch
        # 0 index due to collate function
        frame = frame[0].cuda()
        det_boxes = det_boxes[0].cuda()

        # Track objects in this frame
        timer.tic()
        online_targets = tracker.update(frame_id, frame, det_boxes)
        online_ltwhs = []
        online_boxes = []
        online_scores = []
        online_ids = []
        for t in online_targets:
            online_ltwhs.append(t.ltwh)
            online_boxes.append(t.box.detach().clone().cpu().numpy())
            online_scores.append(t.score)
            online_ids.append(t.tracklet_id)
        timer.toc()

        # Log every 20 frames
        if frame_id % 20 == 0:
            log_or_print(logger, f"Processed frame {frame_id} ({1. / max(1e-5, timer.average_time):.2f} fps)")

        # Store results
        results.append((frame_id, online_ltwhs, online_ids))

        # Visualize tracking results
        if plot_frames:
            cv2_frame = tensor_to_cv2(frame)
            image = plot_boxes(cv2_frame, online_boxes, obj_ids=online_ids,
                scores=online_scores, show_scores=True,
                image_scale=900 / cv2_frame.shape[0], show_info=True,
                frame_id=frame_id, fps=1. / timer.average_time)
            cv2.imshow("Tracking", image)
            while True:
                # Wait for keys
                key = cv2.waitKey(0)
                # Quit, press "q" or "Esc"
                if key in [ord("q"), 27]:
                    exit(0)
                # Next, press "space"
                elif key == 32:
                    break

    # Write files
    write_results(seq_result_file, results, logger)

    # Release GPU memory
    del tracker


def test(
    config,
    logger,
    dataset="MOT17",
    which_set="train",
    public_detection="all",
    sequence="all",
    result_name="TADAM_MOT17",
    evaluation=True,
    plot_frames=False,
):
    # Set directories
    result_folder = os.path.join(config.PATHS.RESULT_ROOT, dataset, result_name)
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)

    full_seq_names, seqs, pds = get_seq_names(dataset, which_set, public_detection, sequence)
    for seq_name, seq, pd in zip(full_seq_names, seqs, pds):
        dataloader = torch.utils.data.DataLoader(
            MOTTracking(config.PATHS.DATASET_ROOT, dataset, which_set, seq, pd),
            batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
        seq_result_file = os.path.join(result_folder, f"{seq_name}.txt")
        log_or_print(logger, f"Sequence: {seq_name}")
        test_sequence(dataloader, config, logger, seq_result_file, plot_frames)

    # Use matlab code to evaluate training set
    if which_set == "train" and evaluation:
        log_or_print(logger, f"Starting Evaluation")
        benchmark(dataset, result_name, config.PATHS.EVAL_ROOT, config.PATHS.RESULT_ROOT, full_seq_names, logger)
        summary = os.path.join(config.PATHS.RESULT_ROOT, dataset, result_name, f"{result_name}_result_summary.txt")
        titles = []
        values = []
        with open(summary, "r") as f:
            titles = f.readline().split()
            values = f.readline().split()
        log_or_print(logger, f"Evaluation Summary")
        for i in range(len(titles) // 10 + 1):
            log_or_print(logger, "\t".join(titles[i * 10: min((i + 1) * 10, len(titles))]))
            log_or_print(logger, "\t".join(values[i * 10: min((i + 1) * 10, len(titles))]))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MOT tracking")
    parser.add_argument("--result-name", default="TADAM_MOT17_train", type=str, help="name for saving results")
    parser.add_argument("--config", default=None, type=str, help="config file to be loaded")
    parser.add_argument("--which_set", default="train", type=str,
        choices=["train", "test"], help="which set to run on")
    parser.add_argument("--public-detection", default="all", choices=["all", "DPM", "FRCNN", "SDP"],
        type=str, help="test on specified public detection, valid for MOT17 only. default is all")
    parser.add_argument("--sequence", default="all", type=str, help="test on specified sequence. default is all")
    parser.add_argument("--evaluation", action="store_true", help="enable evaluation on results. requires matlab")
    parser.add_argument("--plot-frames", action="store_true", help="show frames of tracking")
    parser.add_argument("-v", "--verbose", action="store_true", help="Display details in console log")
    args = parser.parse_args()

    config, cfg_msg = load_config(args.config)
    logger = get_logger(name="global", save_file=True, overwrite_file=True,
        log_dir=os.path.join(config.PATHS.RESULT_ROOT, config.NAMES.DATASET, args.result_name),
        log_name=f"{args.result_name}", console_verbose=args.verbose)
    log_or_print(logger, cfg_msg)

    test(config, logger, dataset=config.NAMES.DATASET, which_set=args.which_set,
        public_detection=args.public_detection, sequence=args.sequence,
        result_name=args.result_name, evaluation=args.evaluation, plot_frames=args.plot_frames)
