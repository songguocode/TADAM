import itertools

import torch
from torchvision.ops import nms, box_iou

from ..modules.detector import Detector
from .tracklet import Tracklet, TrackletList
from .detection import Detection

from ..utils.matching import iou, box_giou, reid_distance, linear_assignment
from ..utils.image_processing import tensor_to_cv2, cmc_align
from ..utils.log import log_or_print


class OnlineTracker(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        # Model, in cuda by default
        self.detector = Detector(num_classes=2, tracking=True,
            config=self.config, logger=self.logger).cuda()

        # Tracklets
        self.active_tracklets = TrackletList("Active")
        self.lost_tracklets = TrackletList("Lost")

        # Record last frame for camera movement compensation
        self.last_frame = None

    def update(self, frame_id, frame, det_boxes):
        # Frame info
        log_or_print(self.logger, f"========== Frame {frame_id:4d} ==========", level="debug")
        frame_height, frame_width = frame.shape[1:]

        # ------------------------------ #
        # ------Process Detections------ #
        # ------------------------------ #

        detections = []
        if len(det_boxes):  # In case no detections available
            det_boxes, det_scores, det_embeddings = \
                self.detector.predict_boxes(
                    frame,
                    det_boxes,
                    prediction_type="detection"
                )

            # Filter low scores
            high_score_keep = torch.ge(det_scores, self.config.TRACKING.MIN_SCORE_DETECTION)
            det_scores = det_scores[high_score_keep]
            det_boxes = det_boxes[high_score_keep]
            det_embeddings = det_embeddings[high_score_keep]

            if len(det_scores):  # in case of empty
                # Apply nms to suppress close detections, especially for DPM detector
                indices = nms(det_boxes, det_scores, self.config.TRACKING.NMS_DETECTION).cpu().numpy()
                det_boxes = det_boxes[indices]
                det_scores = det_scores[indices]
                det_embeddings = det_embeddings[indices]

                if len(det_scores):  # in case of empty
                    # Add as detection objects
                    for index in range(len(det_scores)):
                        detections.append(Detection(det_boxes[index], det_scores[index].item(),
                            embedding=det_embeddings[index].clone().detach().unsqueeze(0)))
        if len(detections):
            log_or_print(self.logger, f"{len(detections)} detections found after filtering", level="debug")

        # ---------------------------- #
        # -------Tracklet Update------ #
        # ---------------------------- #

        # Camera movement compensation
        warp = None
        if self.last_frame is not None:
            warp = cmc_align(self.last_frame, tensor_to_cv2(frame))
            for t in itertools.chain(self.active_tracklets, self.lost_tracklets):
                t.cmc_update(warp)

        # Box outside or overlap with edge. Do twice
        # First time here as CMC could move boxes
        # Second time later after prediction which also move boxes
        if len(self.active_tracklets):
            indices = []
            for i, t in enumerate(self.active_tracklets):
                if t.box[0] > frame_width - self.config.TRACKING.MIN_BOX_SIZE or \
                        t.box[1] > frame_height - self.config.TRACKING.MIN_BOX_SIZE or \
                        t.box[2] < self.config.TRACKING.MIN_BOX_SIZE or \
                        t.box[3] < self.config.TRACKING.MIN_BOX_SIZE:
                    indices.append(i)
            self.move_tracklets("Outside_cmc\t",
                [t for i, t in enumerate(self.active_tracklets) if i in indices],
                self.active_tracklets, self.lost_tracklets)

        # Prediction on active tracklets
        if len(self.active_tracklets):
            # Collect embeddings for target attention
            target_embeddings = torch.cat([t.embedding for i, t in enumerate(self.active_tracklets)])
            target_bools = torch.ones(len(self.active_tracklets), device=target_embeddings.device, dtype=torch.bool)

            # Collect embeddings for distractor attention
            distractor_bools, distractor_embeddings, distractor_ious = \
                self.collect_distractors(
                    self.active_tracklets,
                    self.active_tracklets,  # GS
                    self.config.TRACKING.MIN_OVERLAP_AS_DISTRACTOR
                )

            # Prediction
            t_boxes = torch.stack([t.box for t in self.active_tracklets])
            t_boxes, t_scores, t_embeddings = \
                self.detector.predict_boxes(
                    frame,
                    t_boxes,
                    prediction_type="tracklet",
                    box_ids=torch.tensor([t.tracklet_id for t in self.active_tracklets]).cuda(),
                    target_bools=target_bools,
                    target_embeddings=target_embeddings,
                    distractor_ious=distractor_ious,
                    distractor_bools=distractor_bools,
                    distractor_embeddings=distractor_embeddings
                )

            # Filter by scores
            high_score_keep = torch.ge(t_scores, self.config.TRACKING.MIN_SCORE_ACTIVE_TRACKLET)
            # Update with new position for high score ones
            for i, t in enumerate(self.active_tracklets):
                if high_score_keep[i]:
                    t.update(t_boxes[i, :], t_scores[i].item(),
                        t_embeddings[i].clone().detach().unsqueeze(0))
                    log_or_print(self.logger, f"Updated\t\t{t.tracklet_id}", level="debug")
            # Send low score ones to lost list
            self.move_tracklets("Low class score",
                [t for i, t in enumerate(self.active_tracklets) if not high_score_keep[i]],
                self.active_tracklets, self.lost_tracklets)

        # NMS
        if len(self.active_tracklets):  # In case no active tracklets after filtering
            scores = torch.tensor([t.score for t in self.active_tracklets]).cuda()
            boxes = torch.stack([t.box for t in self.active_tracklets])
            indices = nms(boxes, scores, self.config.TRACKING.NMS_ACTIVE_TRACKLET).cpu().numpy()
            self.move_tracklets("NMS\t",
                [t for i, t in enumerate(self.active_tracklets) if i not in indices],
                self.active_tracklets, self.lost_tracklets)

        # Box outside or overlap with edge, second time
        if len(self.active_tracklets):
            indices = []
            for i, t in enumerate(self.active_tracklets):
                if t.box[0] > frame_width - self.config.TRACKING.MIN_BOX_SIZE or \
                        t.box[1] > frame_height - self.config.TRACKING.MIN_BOX_SIZE or \
                        t.box[2] < self.config.TRACKING.MIN_BOX_SIZE or \
                        t.box[3] < self.config.TRACKING.MIN_BOX_SIZE:
                    indices.append(i)
            self.move_tracklets("Outside_pred",
                [t for i, t in enumerate(self.active_tracklets) if i in indices],
                self.active_tracklets, self.lost_tracklets)

        # Remove ones with too small boxes
        if len(self.active_tracklets):  # In case no active tracklets after filtering
            indices = []
            for i, t in enumerate(self.active_tracklets):
                if t.ltwh[2] < self.config.TRACKING.MIN_BOX_SIZE or t.ltwh[3] < self.config.TRACKING.MIN_BOX_SIZE:
                    indices.append(i)
            self.move_tracklets("Too small\t",
                [t for i, t in enumerate(self.active_tracklets) if i in indices],
                self.active_tracklets, self.lost_tracklets)

        # Remove boxes too close to edge with only a narrow visible region
        # Necessary for MOT16&17, as gt annotation boxes could go outside
        # Trained results follow same pattern, thus leave narrow boxes at edges
        if len(self.active_tracklets):  # In case no active tracklets after filtering
            indices = []
            for i, t in enumerate(self.active_tracklets):
                _, _, w, h = t.ltwh
                min_ratio = 0.25
                if t.box[0] > frame_width - w * min_ratio or \
                        t.box[1] > frame_height - h * min_ratio or \
                        t.box[2] < w * min_ratio or \
                        t.box[3] < h * min_ratio:
                    indices.append(i)
            self.move_tracklets("Edge\t\t",
                [t for i, t in enumerate(self.active_tracklets) if i in indices],
                self.active_tracklets, self.lost_tracklets)

        # -------------------- #
        # ------Matching------ #
        # -------------------- #

        if len(detections):  # In case no detections
            # Check if a detection is covered by an active tracklet, simplified matching
            for det in detections:
                for t in self.active_tracklets:
                    if iou(det.ltwh, t.ltwh) > self.config.TRACKING.NMS_DETECTION:
                        det.brand_new = False
                        break
            detections = [det for det in detections if det.brand_new]

            # Matching between lost tracklets and detections, then recover ones matched
            if len(self.lost_tracklets) and len(detections):  # In case no detections/lost
                # Use ReID as cost
                t_boxes = torch.stack([t.box for t in self.lost_tracklets])
                det_boxes = torch.stack([d.box for d in detections])
                # Use GIoU to prevent matching too far away
                giou_matrix = box_giou(t_boxes, det_boxes).cpu().numpy()
                # Use id embedding similarity as cost
                cost_matrix = reid_distance(self.lost_tracklets, detections)
                # Filter out those GIoU unqualified (set cost to 1)
                cost_matrix[giou_matrix < self.config.TRACKING.MIN_RECOVER_GIOU] = 1.
                # Linear assignment for matching
                matches, _, unassigned_detection_indices = linear_assignment(cost_matrix,
                    threshold=1 - self.config.TRACKING.MIN_RECOVER_SCORE)
                # Matched with a detection
                recover_t_indices = []
                recover_det_indices = []
                for matched_t_index, matched_det_index in matches:
                    t = self.lost_tracklets[matched_t_index]
                    det = detections[matched_det_index]
                    recover_t_indices.append(matched_t_index)
                    recover_det_indices.append(matched_det_index)
                    t.recover(frame_id, det.box, det.embedding.clone().detach())
                if len(recover_t_indices):
                    self.move_tracklets("Recovered\t", [t for i, t in enumerate(self.lost_tracklets)
                        if i in recover_t_indices], self.lost_tracklets, self.active_tracklets)
                # Remove matched detections
                detections = [det for i, det in enumerate(detections) if i not in recover_det_indices]

        # Initiate new tracklets from unmatched new detections
        for i, d in enumerate(detections):
            t = Tracklet(frame_id, d.box, self.config,
                embedding=d.embedding.clone().detach(),
                memory_net=self.detector.memory_net)
            log_or_print(self.logger, f"New\t\t\t{t.tracklet_id}", level="debug")
            self.active_tracklets.append(t)

        # Update info
        for t in self.active_tracklets:
            t.update_active_info(frame_id)
        for t in self.lost_tracklets:
            t.update_lost_info()

        # Remove long lost tracklets
        for t in self.lost_tracklets:
            if t.time_since_update >= t.max_lost_frames:
                log_or_print(self.logger, f"Removed\t{t.tracklet_id}", level="debug")
                self.lost_tracklets.remove(t)
                del t

        log_or_print(self.logger, f"Active Tracklets\t{sorted([t.tracklet_id for t in self.active_tracklets])}", level="debug")
        log_or_print(self.logger, f"Lost Tracklets\t{sorted([t.tracklet_id for t in self.lost_tracklets])}", level="debug")

        self.last_frame = tensor_to_cv2(frame)
        torch.cuda.empty_cache()

        return self.active_tracklets

    def move_tracklets(self, reason, tracklets, source, target):
        for t in tracklets:
            if t in source:
                log_or_print(self.logger, f"{reason}\t{t.tracklet_id}\t{source.name} ==> {target.name}", level="debug")
                source.remove(t)
                target.append(t)

    # Potential distractors input includes target themselves
    def collect_distractors(self, target_tracklets, distractor_tracklets, min_distractor_overlap=0.2):
        zero_bools = torch.zeros(len(target_tracklets),
                device=target_tracklets[0].box.device, dtype=torch.bool)
        zero_embeddings = torch.zeros([len(distractor_tracklets)] + list(target_tracklets[0].embedding.size()[1:]),
                device=target_tracklets[0].embedding.device, dtype=target_tracklets[0].embedding.dtype)
        zero_ious = torch.zeros(len(distractor_tracklets),
                device=target_tracklets[0].box.device, dtype=target_tracklets[0].box.dtype)
        # No need to compute if only one tracklet exists
        if len(distractor_tracklets) < 2:
            return zero_bools, zero_embeddings, zero_ious
        # Match by IoU
        t_boxes = torch.stack([t.box for t in target_tracklets])
        d_boxes = torch.stack([d.box for d in distractor_tracklets])
        iou_matrix = box_iou(t_boxes, d_boxes)
        top_vals, top_matches = iou_matrix.topk(k=2, dim=1)
        # Distractors are ones with second largest IoU
        # As we pass same lists for target and distractor, the one with largest IoU is always itself
        distractor_ious = top_vals[:, 1]
        distractor_bools = top_vals[:, 1] >= min_distractor_overlap
        distractor_indices = top_matches[:, 1][distractor_bools]
        distractor_embeddings = []
        for d_index in distractor_indices:
            distractor_embeddings.append(distractor_tracklets[d_index.item()].embedding)
        if len(distractor_embeddings):
            distractor_embeddings = torch.cat(distractor_embeddings)
            return distractor_bools, distractor_embeddings, distractor_ious
        # If all distractors do no qualify the min_distractor_overlap
        else:
            return zero_bools, zero_embeddings, distractor_ious

    # Release GPU memory
    def __del__(self):
        if "active_tracklets" in self.__dict__:
            del self.active_tracklets
        if "lost_tracklets" in self.__dict__:
            del self.lost_tracklets
        torch.cuda.empty_cache()
