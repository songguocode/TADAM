import numpy as np
import torch

from ..utils.kalman_filter import KalmanFilter
from ..modules.memory import Memory


class TrackletList(list):
    def __init__(self, name):
        self.name = name


class Tracklet(object):
    # Shared among all instances, not accessible outside
    _count = 0

    def __init__(
        self,
        frame_id,
        box,
        config,
        memory_net=None,
        embedding=None,
        training=False,
    ):
        # Initialization with default
        self.tracklet_id = self.next_id()
        self.tracklet_len = 0
        self.time_since_update = 0
        self.score = 1.0
        self.frame_id = frame_id
        self.start_frame = frame_id

        # Position
        self.box = box.detach()

        # Kalman filter
        self.kalman_filter = KalmanFilter(dim=4)
        self.kalman_filter.initiate(self.box.cpu().numpy())

        # Config
        self.config = config

        # Memory aggregation
        assert memory_net is not None, "MemoryNet not passed"
        self.memory_net = memory_net
        if training:
            self.memory_train_input = [embedding]
        self.memory = Memory(embedding, memory_net)

        # Lost properties
        self.max_lost_frames = config.TRACKING.MAX_LOST_FRAMES_BEFORE_REMOVE

        # For training
        self.train_update_count = 0

    @staticmethod
    def next_id():
        Tracklet._count += 1
        return Tracklet._count

    @property
    def ltwh(self):
        # Retrieve left, top, width, height from box (x1y1x2y2)
        ltwh = np.asarray(self.box.clone().detach().cpu().numpy())
        ltwh[2:] -= ltwh[:2]
        return ltwh

    def cmc_update(self, warp):
        """
            warp: affine transform matrix, np.array or None (no ECC)
        """
        warp_tensor = torch.tensor(warp, dtype=self.box.dtype, device=self.box.device)
        p1 = torch.tensor([self.box[0], self.box[1], 1], dtype=self.box.dtype, device=self.box.device).view(3, 1)
        p2 = torch.tensor([self.box[2], self.box[3], 1], dtype=self.box.dtype, device=self.box.device).view(3, 1)
        p1_n = torch.mm(warp_tensor, p1).view(1, 2)
        p2_n = torch.mm(warp_tensor, p2).view(1, 2)
        box = torch.cat((p1_n, p2_n), 1).view(1, -1).squeeze(0)
        self.update(box)

    def update(self, box, score=None, embedding=None):
        # Prevent abnormal aspect ratios
        aspect_ratio = (box[3] - box[1]) / (box[2] - box[0])
        # Keep center unchanged, use original width & height
        if aspect_ratio < 1.0 or aspect_ratio > 4.0:
            cx = box[0] + (box[2] - box[0]) / 2
            cy = box[1] + (box[3] - box[1]) / 2
            w = self.box[2] - self.box[0]
            h = self.box[3] - self.box[1]
            box = torch.tensor([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                dtype=self.box.dtype, device=self.box.device)
        self.box = box

        # Update kalman filter
        self.kalman_filter.predict()
        self.kalman_filter.update(box.cpu().numpy())

        if score is not None:
            self.score = score
        if embedding is not None:
            self.update_embedding(embedding)

    def predict(self):
        self.box = torch.tensor(self.kalman_filter.predict(), dtype=self.box.dtype, device=self.box.device)

    def recover(self, frame_id, box, embedding):
        self.frame_id = frame_id
        self.box = box.detach()
        self.time_since_update = 0
        self.update_embedding(embedding)

        # Reset prediction
        self.kalman_filter.initiate(self.box.cpu().numpy())

        # Reset score
        self.score = 1.0

    @property
    def avg_embedding(self):
        return self.embedding.mean(-1).mean(-1)

    @property
    def embedding(self):
        return self.memory.h_state

    def update_embedding(self, new_embedding, training=False):
        # Reset for training when updated for 4 times
        if training and self.train_update_count >= 4:
            self.memory = Memory(new_embedding, self.memory_net)
            self.memory_train_input = [new_embedding]
            self.train_update_count = 0
        else:
            # For tracking
            if not training:
                self.memory.update(new_embedding, self.memory_net)
            # For training
            else:
                self.memory_train_input.append(new_embedding)
                self.memory.train_update(self.memory_train_input, self.memory_net)
                self.train_update_count += 1

    def update_active_info(self, frame_id):
        # Except for new tracklet
        if frame_id > self.frame_id:
            self.frame_id = frame_id
            self.time_since_update = 0
            self.tracklet_len += 1

    def update_lost_info(self):
        self.time_since_update += 1

    def __del__(self):
        if "memory" in self.__dict__:
            del self.memory
        # Release GPU memory
        torch.cuda.empty_cache()
