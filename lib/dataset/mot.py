import configparser
import csv
import os
import itertools

from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor


class MOTDetection(torch.utils.data.Dataset):
    """
        Data class for detection
        Loads all images in all sequences at once
        To be used for training
    """

    def __init__(
        self,
        root="../datasets/",
        dataset="MOT17Det",
        transforms=None,
        vis_threshold=0.1,
    ):
        predefined_datasets = ["MOT16", "MOT17Det", "MOT20"]
        assert dataset in predefined_datasets, \
            f"Provided dataset name '{dataset}' is not in predefined datasets: {predefined_datasets}"

        self.root = os.path.join(root, dataset, "train")
        self.transforms = transforms
        self._vis_threshold = vis_threshold
        self._classes = ("__background__", "pedestrian")
        self._global_id_counter = 0
        self._local_to_global_dict = {}
        self._global_to_local_dict = {}
        self._img_paths = []
        self._aspect_ratios = []

        for f in sorted(os.listdir(self.root)):
            path = os.path.join(self.root, f)
            config_file = os.path.join(path, "seqinfo.ini")

            assert os.path.exists(config_file), f"Path does not exist: {config_file}"

            config = configparser.ConfigParser()
            config.read(config_file)
            seq_len = int(config["Sequence"]["seqLength"])
            im_width = int(config["Sequence"]["imWidth"])
            im_height = int(config["Sequence"]["imHeight"])
            im_ext = config["Sequence"]["imExt"]
            im_dir = config["Sequence"]["imDir"]

            _imDir = os.path.join(path, im_dir)
            aspect_ratio = im_width / im_height

            # Collect global gt_id
            self.process_ids(path)

            for i in range(1, seq_len + 1):
                img_path = os.path.join(_imDir, f"{i:06d}{im_ext}")
                assert os.path.exists(img_path), \
                    "Path does not exist: {img_path}"
                self._img_paths.append(img_path)
                self._aspect_ratios.append(aspect_ratio)

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def num_ids(self):
        return self._global_id_counter

    def _get_annotation(self, idx):
        """
            Obtain annotation from gt file
        """

        img_path = self._img_paths[idx]
        file_index = int(os.path.basename(img_path).split(".")[0])

        gt_file = os.path.join(os.path.dirname(
            os.path.dirname(img_path)), "gt", "gt.txt")
        seq_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))

        assert os.path.exists(gt_file), f"GT file does not exist: {gt_file}"

        bounding_boxes = []

        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=",")
            for row in reader:
                visibility = float(row[8])
                local_id = f"{seq_name}-{int(row[1])}"
                if int(row[0]) == file_index and int(row[6]) == 1 and int(row[7]) == 1 and \
                        visibility > self._vis_threshold:
                    bb = {}
                    bb["gt_id"] = self._local_to_global_dict[local_id]
                    bb["bb_left"] = int(row[2])
                    bb["bb_top"] = int(row[3])
                    bb["bb_width"] = int(row[4])
                    bb["bb_height"] = int(row[5])
                    bb["visibility"] = visibility

                    bounding_boxes.append(bb)

        num_objs = len(bounding_boxes)

        boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
        visibilities = torch.zeros((num_objs), dtype=torch.float32)
        gt_ids = torch.zeros((num_objs), dtype=torch.int64)

        for i, bb in enumerate(bounding_boxes):
            x1 = bb["bb_left"]  # GS
            y1 = bb["bb_top"]
            x2 = x1 + bb["bb_width"]
            y2 = y1 + bb["bb_height"]
            boxes[i, 0] = x1
            boxes[i, 1] = y1
            boxes[i, 2] = x2
            boxes[i, 3] = y2
            visibilities[i] = bb["visibility"]
            gt_ids[i] = bb["gt_id"]

        return {"boxes": boxes,
                "labels": torch.ones((num_objs,), dtype=torch.int64),
                "image_id": torch.tensor([idx]),
                "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                "iscrowd": torch.zeros((num_objs,), dtype=torch.int64),
                "visibilities": visibilities,
                "frame_id": torch.tensor([file_index]),
                "gt_ids": gt_ids}

    def process_ids(self, path):
        """
            Global id is 0-based, indexed across all sequences
            All ids are considered, regardless of used or not
        """
        seq_name = os.path.basename(path)
        if seq_name not in self._global_to_local_dict.keys():
            self._global_to_local_dict[seq_name] = {}
        gt_file = os.path.join(path, "gt", "gt.txt")
        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=",")
            for row in reader:
                local_id = f"{seq_name}-{int(row[1])}"
                if int(row[6]) == 1 and int(row[7]) == 1:
                    if local_id not in self._local_to_global_dict.keys():
                        self._local_to_global_dict[local_id] = self._global_id_counter
                        self._global_to_local_dict[seq_name][self._global_id_counter] = int(row[1])
                        self._global_id_counter += 1

    def __getitem__(self, idx):
        # Load image
        img_path = self._img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        # Get annotation
        target = self._get_annotation(idx)
        # Apply augmentation transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self._img_paths)


class MOTTracking(torch.utils.data.Dataset):
    """
        Data class for tracking
        Loads one sequence at a time
        To be used for tracking
    """

    def __init__(
        self,
        root="../datasets/",
        dataset="MOT17",
        which_set="train",
        sequence="02",
        public_detection="None",
        vis_threshold=0.1,
    ):
        # Check dataset
        predefined_datasets = ["MOT16", "MOT17", "MOT20"]
        assert dataset in predefined_datasets, \
            f"Provided dataset name '{dataset}' is not in predefined datasets: {predefined_datasets}"
        # Different public detections for MOT17
        if dataset == "MOT17":
            assert public_detection in ["DPM", "FRCNN", "SDP"], "Incorrect public detection provided"
            public_detection = f"-{public_detection}"
        # No public detection names for MOT16 and MOT20
        else:
            assert public_detection == "None", f"No public detection should be provided for {dataset}"
            public_detection = ""
        # Check train/test
        assert which_set in ["train", "test"], "Invalid choice between 'train' and 'test'"
        # Check sequence, convert to two-digits string format
        assert sequence.isdigit(), "Non-digit sequence provided"
        sequence = f"{int(sequence):02d}"
        dict_sequences = {
            "MOT16": {
                "train": ["02", "04", "05", "09", "10", "11", "13"],
                "test": ["01", "03", "06", "07", "08", "12", "14"],
            },
            "MOT17": {
                "train": ["02", "04", "05", "09", "10", "11", "13"],
                "test": ["01", "03", "06", "07", "08", "12", "14"],
            },
            "MOT20": {
                "train": ["01", "02", "03", "05"],
                "test": ["04", "06", "07", "08"],
            }
        }
        assert sequence in dict_sequences[dataset][which_set], \
            f"Sequence for {dataset}/{which_set} must be in [{dict_sequences[dataset][which_set]}]"

        self._img_paths = []
        self._vis_threshold = vis_threshold

        # Load images
        self.path = os.path.join(root, dataset, which_set, f"{dataset}-{sequence}{public_detection}")
        config_file = os.path.join(self.path, "seqinfo.ini")

        assert os.path.exists(config_file), f"Path does not exist: {config_file}"

        config = configparser.ConfigParser()
        config.read(config_file)
        seq_len = int(config["Sequence"]["seqLength"])
        im_ext = config["Sequence"]["imExt"]
        im_dir = config["Sequence"]["imDir"]

        _imDir = os.path.join(self.path, im_dir)

        for i in range(1, seq_len + 1):
            img_path = os.path.join(_imDir, f"{i:06d}{im_ext}")
            assert os.path.exists(img_path), \
                "Path does not exist: {img_path}"
            self._img_paths.append(img_path)

    def _get_annotation(self, idx):
        """
            Obtain annotation for detections (train/test) and ground truths (train only)
        """
        img_path = self._img_paths[idx]
        file_index = int(os.path.basename(img_path).split(".")[0])

        det_file = os.path.join(os.path.dirname(
            os.path.dirname(img_path)), "det", "det.txt")
        assert os.path.exists(det_file), \
            f"Det file does not exist: {det_file}"
        det_boxes, _, det_scores, _ = read_mot_file(det_file, file_index, self._vis_threshold, is_gt=False)

        # No GT for test set
        if "test" in self.path:
            return det_boxes, None, None, None, None

        gt_file = os.path.join(os.path.dirname(
            os.path.dirname(img_path)), "gt", "gt.txt")
        assert os.path.exists(gt_file), \
            f"GT file does not exist: {gt_file}"
        gt_boxes, gt_ids, _, gt_visibilities = read_mot_file(gt_file, file_index, self._vis_threshold, is_gt=True)

        return det_boxes, det_scores, gt_boxes, gt_ids, gt_visibilities

    def __getitem__(self, idx):
        # Load image
        img_path = self._img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = to_tensor(img)
        # Get annotation
        det_boxes, det_scores, gt_boxes, gt_ids, gt_visibilities = self._get_annotation(idx)

        return img, det_boxes, det_scores, gt_boxes, gt_ids, gt_visibilities

    def __len__(self):
        return len(self._img_paths)


def read_mot_file(file, file_index, vis_threshold=0.1, is_gt=False):
    """
        Read data from mot files, gt or det or tracking result
    """
    bounding_boxes = []
    with open(file, "r") as inf:
        reader = csv.reader(inf, delimiter=",")
        for row in reader:
            visibility = float(row[8]) if is_gt else -1.0
            if int(row[0]) == file_index and \
                    ((is_gt and (int(row[6]) == 1 and int(row[7]) == 1 and visibility > vis_threshold)) or
                    not is_gt):  # Only requires class=pedestrian and confidence=1 for gt
                bb = {}
                bb["gt_id"] = int(row[1])
                bb["bb_left"] = float(row[2])
                bb["bb_top"] = float(row[3])
                bb["bb_width"] = float(row[4])
                bb["bb_height"] = float(row[5])
                bb["bb_score"] = float(row[6]) if not is_gt else 1
                bb["visibility"] = visibility
                bounding_boxes.append(bb)

    num_objs = len(bounding_boxes)
    boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
    scores = torch.zeros((num_objs), dtype=torch.float32)
    visibilities = torch.zeros((num_objs), dtype=torch.float32)
    ids = torch.zeros((num_objs), dtype=torch.int64)
    for i, bb in enumerate(bounding_boxes):
        x1 = bb["bb_left"]  # GS
        y1 = bb["bb_top"]
        x2 = x1 + bb["bb_width"]
        y2 = y1 + bb["bb_height"]
        boxes[i, 0] = x1
        boxes[i, 1] = y1
        boxes[i, 2] = x2
        boxes[i, 3] = y2
        scores[i] = bb["bb_score"]
        visibilities[i] = bb["visibility"]
        ids[i] = bb["gt_id"]

    return boxes, ids, scores, visibilities


def collate_fn(batch):
    """
        Function for dataloader
    """
    return tuple(zip(*batch))


def get_seq_names(dataset, which_set, public_detection, sequence):
    """
        Get name of all required sequences
    """
    # Process inputs
    if public_detection == "all":
        if dataset == "MOT17":
            public_detection_list = ["DPM", "FRCNN", "SDP"]
        else:
            public_detection_list = ["None"]
    else:
        public_detection_list = [public_detection]

    if sequence == "all":
        if dataset == "MOT20":
            if which_set == "train":
                sequence_list = ["01", "02", "03", "05"]
            else:
                sequence_list = ["04", "06", "07", "08"]
        else:
            if which_set == "train":
                sequence_list = ["02", "04", "05", "09", "10", "11", "13"]
            else:
                sequence_list = ["01", "03", "06", "07", "08", "12", "14"]
    else:
        sequence_list = [sequence]
    # Iterate through all sequences
    full_names = []
    seqs = []
    pds = []  # public detections for each sequence
    for pd, seq in list(itertools.product(public_detection_list, sequence_list)):
        seqs.append(seq)
        pd_suffix = f"-{pd}" if dataset == "MOT17" else ""
        pds.append(pd)
        curr_seq = f"{dataset}-{seq}{pd_suffix}"
        full_names.append(curr_seq)
    return full_names, seqs, pds
