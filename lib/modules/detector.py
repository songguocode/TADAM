import os
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch.autograd.function import Function
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes
from torchvision.ops import MultiScaleRoIAlign, box_iou
# Local files
from .faster_rcnn import FasterRCNN, TwoMLPHead, FastRCNNPredictor
from .roi_heads import fastrcnn_loss
from .identity import IDModule
from .memory import MemoryNet
from .attention import NonLocalAttention
from .integration import IntegrationModule
from ..utils.model_loader import load_model
from ..tracking.tracklet import Tracklet


class Detector(FasterRCNN):
    def __init__(
        self,
        config,
        num_classes=2,
        num_ids=1000,
        tracking=False,
        logger=None,
    ):
        super(Detector, self).__init__(
            resnet_fpn_backbone(config.NAMES.BACKBONE, False),
            num_classes=num_classes
        )
        assert config is not None, "Config not passed"
        self.config = config

        # Load components
        self.memory_net = MemoryNet(
            feature_size=(256, 7, 7),
            num_ids=num_ids,
            kernel_size=(3, 3),
            bias=True
        )
        self.roi_heads.id_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2
        )
        self.roi_heads.id_module = IDModule(
            in_channels=256,
            out_channels=256,
            num_ids=num_ids
        )
        self.roi_heads.target_enhance = NonLocalAttention(
            in_channels=256,
            inter_channels=128
        )
        self.roi_heads.distractor_reduce = NonLocalAttention(
            in_channels=256,
            inter_channels=128
        )
        self.roi_heads.integration = IntegrationModule(min_iou=0.2)
        self.roi_heads.hard_box_head = TwoMLPHead(
            self.roi_heads.box_head.fc6.in_features,
            self.roi_heads.box_head.fc6.out_features
        )
        self.roi_heads.hard_box_predictor = FastRCNNPredictor(
            self.roi_heads.box_predictor.cls_score.in_features,
            num_classes=num_classes
        )

        # Load trained model for tracking
        if tracking:
            model_path = os.path.join(
                self.config.PATHS.MODEL_ROOT,
                self.config.NAMES.MODEL
            )
            self = load_model(self, model_path, logger)
            # Freeze model to speed up inferencing
            for param in self.parameters():
                param.requires_grad = False
        # Load checkpoint for training
        else:
            checkpoint_path = os.path.join(
                self.config.PATHS.MODEL_ROOT,
                self.config.NAMES.CHECKPOINT
            )
            self = load_model(self, checkpoint_path, logger)

    def predict_boxes(
        self,
        frame,
        boxes,
        prediction_type="detection",
        distractor_ious=None,
        box_ids=None,
        target_bools=None,
        target_embeddings=None,
        distractor_bools=None,
        distractor_embeddings=None
    ):
        """
            Make predictions from given bounding boxes
            Either from public detections (basic), or from tracked targets (with TADA)
        """
        device = list(self.parameters())[0].device
        images = frame.unsqueeze(0)
        images = images.to(device)
        boxes.to(device)

        targets = None
        original_image_sizes = [images.shape[-2:]]
        # Image and box sizes are changed inside the RCNN transform
        images, targets = self.transform(images, targets)

        backbone_features = self.backbone(images.tensors)
        if isinstance(backbone_features, torch.Tensor):
            backbone_features = OrderedDict([(0, backbone_features)])

        # Resize to transformed size
        proposals = [resize_boxes(boxes, original_image_sizes[0], images.image_sizes[0])]

        # Get box features by pooling
        box_features = self.roi_heads.box_roi_pool(backbone_features, proposals, images.image_sizes)

        # Basic prediction for given public detections
        assert prediction_type in ["detection", "tracklet"], "Invalid prediction type"
        if prediction_type == "detection":
            box_features = self.roi_heads.box_head(box_features)
            class_logits, box_regression = self.roi_heads.box_predictor(box_features)
        # Target-aware and distractor-aware for tracked targets
        else:
            # Use distractor iou threshold to choose between easy/hard classifier
            awareness_bool = distractor_ious > self.config.TRACKING.MIN_OVERLAP_AS_DISTRACTOR
            # No awareness for easy cases
            easy_box_features = self.roi_heads.box_head(box_features[~awareness_bool])
            easy_class_logits, easy_box_regression = self.roi_heads.box_predictor(easy_box_features)
            # Apply awareness for hard cases
            if len(target_bools[awareness_bool]):  # In case no hard cases
                hard_awareness, _ = self.process_id_embeddings(
                    [p[awareness_bool] for p in proposals],
                    backbone_features=backbone_features,
                    image_shapes=images.image_sizes,
                    boxes_type="tracklet",
                    purpose="awareness",
                    target_bools=target_bools[awareness_bool],
                    target_embeddings=target_embeddings[awareness_bool],
                    distractor_ious=distractor_ious[awareness_bool],
                    distractor_bools=distractor_bools[awareness_bool],
                    distractor_embeddings=distractor_embeddings
                )
                hard_input_features = box_features[awareness_bool] + hard_awareness
                hard_box_features = self.roi_heads.hard_box_head(hard_input_features)
                hard_class_logits, hard_box_regression = self.roi_heads.hard_box_predictor(hard_box_features)
            # Combine both cases, create empty data first
            class_logits = torch.zeros([len(boxes)] + list(easy_class_logits.size()[1:]),
                dtype=easy_class_logits.dtype, device=easy_class_logits.device)
            box_regression = torch.zeros([len(boxes)] + list(easy_box_regression.size()[1:]),
                dtype=easy_box_regression.dtype, device=easy_box_regression.device)
            # Assign values according to awareness_bool
            class_logits[~awareness_bool] = easy_class_logits
            box_regression[~awareness_bool] = easy_box_regression
            if len(target_bools[awareness_bool]):  # In case no hard cases
                class_logits[awareness_bool] = hard_class_logits
                box_regression[awareness_bool] = hard_box_regression

        # Process predictions
        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        # Get pedestrian class
        pred_boxes = pred_boxes[:, 1]
        pred_scores = pred_scores[:, 1]

        # Recover original size
        pred_boxes_orig = resize_boxes(pred_boxes, images.image_sizes[0], original_image_sizes[0])

        # Output
        id_embeddings, _ = self.process_id_embeddings(
            [pred_boxes],
            backbone_features=backbone_features,
            image_shapes=images.image_sizes,
            boxes_type=prediction_type,
            purpose="embedding",
            target_bools=target_bools,
            target_embeddings=target_embeddings,
            distractor_ious=distractor_ious,
            distractor_bools=distractor_bools,
            distractor_embeddings=distractor_embeddings)

        return pred_boxes_orig, pred_scores, id_embeddings

    def custom_train(
        self,
        images,
        targets,
        warmup=False
    ):
        """
            TADAM training
        """
        device = list(self.parameters())[0].device
        if targets is None:
            raise ValueError("In training, targets should be passed")

        # Remove image and target with less than one ground truth, happens in MOT16/17-05
        non_empty_indices = []
        for i, t in enumerate(targets):
            if len(t['boxes']) > 1:
                non_empty_indices.append(i)
        images = [img for i, img in enumerate(images) if i in non_empty_indices]
        targets = [t for i, t in enumerate(targets) if i in non_empty_indices]

        # Extract backbone features
        images, targets = self.transform(images, targets)
        backbone_features = self.backbone(images.tensors)
        if isinstance(backbone_features, torch.Tensor):
            backbone_features = OrderedDict([('0', backbone_features)])

        # Losses
        all_losses = {}

        # ====== RPN ====== #
        # Only works in training to select RoIs, not used in tracking
        proposals, proposal_losses = self.rpn(images, backbone_features, targets)
        self.update_losses(all_losses, proposal_losses)

        # ====== Box Features ====== #
        if targets is not None:
            for t in targets:
                assert t["boxes"].dtype in (torch.float, torch.double, torch.half), 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
        proposals, _, labels, regression_targets = self.roi_heads.select_training_samples(proposals, targets)
        assert labels is not None and regression_targets is not None, "Invalid labels/regression_targets"

        # Extract box features from images
        # box_features from different images are concatenated if input has multiple images
        box_features = self.roi_heads.box_roi_pool(backbone_features, proposals, images.image_sizes)

        # ====== Easy cases without identity awareness ====== #
        if not warmup:
            # Easy cases without awareness
            easy_box_features = self.roi_heads.box_head(box_features)
            easy_class_logits, easy_box_regression = self.roi_heads.box_predictor(easy_box_features)
            # Losses
            easy_box_losses = {}
            loss_easy_classifier, loss_easy_box_reg = fastrcnn_loss(
                easy_class_logits, easy_box_regression, labels, regression_targets)
            easy_box_losses = {
                "loss_classifier": loss_easy_classifier,
                "loss_box_reg": loss_easy_box_reg,
            }
            self.update_losses(all_losses, easy_box_losses)
            # Train hard case head & predictor with easy cases for basic abilities
            hard_basic_box_features = self.roi_heads.hard_box_head(box_features)
            hard_basic_class_logits, hard_basic_box_regression = self.roi_heads.hard_box_predictor(hard_basic_box_features)
            # Losses
            hard_basic_box_losses = {}
            loss_hard_basic_classifier, loss_hard_basic_box_reg = fastrcnn_loss(
                hard_basic_class_logits, hard_basic_box_regression, labels, regression_targets)
            hard_basic_box_losses = {
                "loss_hard_basic_classifier": loss_hard_basic_classifier,
                "loss_hard_basic_box_reg": loss_hard_basic_box_reg,
            }
            self.update_losses(all_losses, hard_basic_box_losses)

        # ====== Identity training on memory and embedding extraction ====== #
        # Obtain id embeddings for gt boxes, no loss in this step
        # Only gt boxes participates in identity training for accuracy
        gt_id_embeddings, _ = self.process_id_embeddings(
            [t['boxes'] for t in targets],
            backbone_features=backbone_features,
            image_shapes=images.image_sizes,
            boxes_type='detection',
            purpose='embedding'
        )
        # Update tracklets
        cat_boxes = torch.cat([t['boxes'] for t in targets])
        cat_gt_ids = torch.cat([t['gt_ids'] for t in targets])
        memory_losses = []
        for i, (box, gt_id, e) in enumerate(zip(cat_boxes, cat_gt_ids, gt_id_embeddings)):
            # Update an existing tracklet. Each tracklet is reset after certain updates
            if gt_id.item() in self.all_tracklets_dict.keys():
                tracklet = self.all_tracklets_dict[gt_id.item()]
                tracklet.update_embedding(e.detach().unsqueeze(0), training=True)
                # Get loss for memory upon update, using identity loss as supervision
                memory_losses.append(self.memory_net.memory_loss(tracklet.embedding, gt_id.unsqueeze(0)))
            # Initialize a tracklet
            else:
                self.all_tracklets_dict[gt_id.item()] = Tracklet(-1, box, self.config,
                    embedding=e.detach().unsqueeze(0),
                    memory_net=self.memory_net, training=True)
                self.all_tracklets_dict[gt_id.item()].tracklet_id = gt_id.item()
        # Loss for memory
        if len(memory_losses):
            # Average over tracklets in the frame
            loss_memory = sum(memory_losses) / len(memory_losses)
        else:
            loss_memory = torch.tensor(0.0, dtype=gt_id_embeddings.dtype, device=device)
        self.update_losses(all_losses, {
            'loss_mem': loss_memory
        })
        # Loss for embedding extraction
        pos_gt_embeddings, neg_gt_embeddings = self.collect_triplet(cat_gt_ids)
        _, gt_id_losses = self.process_id_embeddings(
            [t['boxes'] for t in targets],
            backbone_features=backbone_features,
            image_shapes=images.image_sizes,
            boxes_type='detection',
            purpose='embedding',
            matched_bools=torch.ones_like(cat_gt_ids).bool(),
            matched_gt_ids=cat_gt_ids,
            pos_id_embeddings=pos_gt_embeddings,
            neg_id_embeddings=neg_gt_embeddings)
        # Reduce identity losses by a ratio after warmup, to balance different losses
        self.update_losses(all_losses, {
            'loss_gt_id_crossentropy':
                gt_id_losses['loss_id_crossentropy'] if warmup
                else gt_id_losses['loss_id_crossentropy'] * self.config.TRAINING.ID_LOSS_RATIO
        })
        self.update_losses(all_losses, {
            'loss_gt_id_triplet':
                gt_id_losses['loss_id_triplet'] if warmup
                else gt_id_losses['loss_id_triplet'] * self.config.TRAINING.ID_LOSS_RATIO
        })

        # ====== Hard cases with identity awareness ====== #
        if not warmup:
            target_bools, target_embeddings, distractor_bools, distractor_embeddings, distractor_ious = \
                self.collect_attention_embeddings(proposals, targets, min_overlap=0.5,
                    min_distractor_overlap=self.config.TRACKING.MIN_OVERLAP_AS_DISTRACTOR)
            id_awareness, _ = self.process_id_embeddings(
                proposals,
                backbone_features=backbone_features,
                image_shapes=images.image_sizes,
                boxes_type='tracklet',
                purpose='awareness',
                target_bools=target_bools,
                target_embeddings=target_embeddings,
                distractor_ious=distractor_ious,
                distractor_bools=distractor_bools,
                distractor_embeddings=distractor_embeddings)
            # Collect box features
            input_features = box_features + id_awareness
            hard_awareness_box_features = self.roi_heads.hard_box_head(input_features)
            hard_awareness_class_logits, hard_awareness_box_regression = \
                self.roi_heads.hard_box_predictor(hard_awareness_box_features)
            # Losses
            loss_hard_awareness_classifier, loss_hard_awareness_box_reg = fastrcnn_loss(
                hard_awareness_class_logits, hard_awareness_box_regression, labels, regression_targets)
            all_losses.update({
                'loss_hard_awareness_classifier': loss_hard_awareness_classifier,
                'loss_hard_awareness_box_reg': loss_hard_awareness_box_reg,
            })

        return all_losses

    def update_losses(self, all_losses, new_loss):
        for key, value in new_loss.items():
            if key in all_losses:
                all_losses[key] += value
            else:
                all_losses.update({key: value})

    def process_id_embeddings(
        self,
        boxes,
        backbone_features=None,
        image_shapes=None,
        boxes_type="detection",
        purpose="embedding",
        matched_bools=None,
        matched_gt_ids=None,
        pos_id_embeddings=None,
        neg_id_embeddings=None,
        target_bools=None,
        target_embeddings=None,
        distractor_ious=None,
        distractor_bools=None,
        distractor_embeddings=None
    ):
        """
            Outputs identity embeddings and/or awareness
            For public detection: extracts identity embeddinng only
            For tracked targets: extract identity embedding + awareness
            For awareness computation: awareness only
        """
        assert purpose in ["embedding", "awareness"], "Invalid purpose"
        id_losses = {}
        assert backbone_features is not None and image_shapes is not None
        id_embeddings = self.roi_heads.id_module(self.roi_heads.id_roi_pool(backbone_features, boxes, image_shapes))
        # ID losses
        if self.training and matched_bools is not None and matched_gt_ids is not None:
            # Cross entropy loss # GS
            loss_id_crossentropy = self.roi_heads.id_module.cross_entropy_loss(
                id_embeddings[matched_bools], matched_gt_ids)
            id_losses.update({"loss_id_crossentropy": loss_id_crossentropy})
            # Triplet loss # GS
            assert pos_id_embeddings is not None
            assert neg_id_embeddings is not None
            assert len(matched_gt_ids) == len(pos_id_embeddings)
            assert len(matched_gt_ids) == len(neg_id_embeddings)
            loss_id_triplet = self.roi_heads.id_module.triplet_loss(
                id_embeddings[matched_bools],
                pos_id_embeddings,
                neg_id_embeddings,
                margin=0.3)
            id_losses.update({"loss_id_triplet": loss_id_triplet})
        # Apply awareness for tracked targets
        enhancement = torch.zeros_like(id_embeddings)
        reduction = torch.zeros_like(id_embeddings)
        if boxes_type == "tracklet":
            # Target
            assert target_bools is not None and target_embeddings is not None
            if len(target_bools):
                enhancement[target_bools] = self.roi_heads.target_enhance(id_embeddings[target_bools].detach(), target_embeddings)
            # Distractor. It is possible to have no distractors
            if distractor_bools is not None and distractor_embeddings is not None and len(distractor_bools) and torch.sum(distractor_bools).item() > 0:
                reduction[distractor_bools] = self.roi_heads.distractor_reduce(id_embeddings[distractor_bools].detach(), distractor_embeddings)
                # Scale up gradient
                reduction[distractor_bools] = _ScaleGradient.apply(reduction[distractor_bools], 2.0)
        # Combine
        awareness = torch.zeros_like(id_embeddings)
        if boxes_type == "tracklet":
            awareness = self.roi_heads.integration(enhancement, reduction, overlaps=distractor_ious)
        # Output
        output = \
            id_embeddings * float(purpose == "embedding") + awareness
        return output, id_losses

    def collect_triplet(self, pos_ids):
        """
            Retrieve pos and neg for triplet in identity training
        """
        pos_embeddings = []
        for p_id in pos_ids:
            t = self.all_tracklets_dict[p_id.item()]
            pos_embeddings.append(t.embedding.detach())
        pos_embeddings = torch.cat(pos_embeddings)
        # Generate random neg indices
        neg_embeddings = []
        neg_gt_ids = torch.tensor(list(self.all_tracklets_dict.keys()), device=pos_ids.device).unsqueeze(0)
        neg_gt_ids_repeated = neg_gt_ids.repeat(pos_ids.size(0), 1)
        neg_gt_ids = neg_gt_ids_repeated[neg_gt_ids_repeated != pos_ids.unsqueeze(1)].reshape(pos_ids.size(0), neg_gt_ids.size(1) - 1)
        neg_selections = torch.randint(0, neg_gt_ids.size(1), (neg_gt_ids.size(0), 1), device=neg_gt_ids.device)
        neg_gt_ids = neg_gt_ids.gather(dim=1, index=neg_selections).squeeze(-1)
        for i, p_id in enumerate(pos_ids):
            t = self.all_tracklets_dict[neg_gt_ids[i].item()]
            neg_embeddings.append(t.embedding.detach())
        neg_embeddings = torch.cat(neg_embeddings)
        return pos_embeddings, neg_embeddings

    def collect_attention_embeddings(
        self,
        proposals,
        targets,
        min_overlap=0.5,
        min_distractor_overlap=0.2
    ):
        """
            Retrieve target reference and distractor reference
        """
        target_bools, target_ref_ids, distractor_bools, distractor_ref_ids, distractor_ious = \
            self.attention_training_match_ids(proposals, targets,
                min_overlap=min_overlap, min_distractor_overlap=min_distractor_overlap)
        target_embeddings = []
        for target_id in target_ref_ids:
            target_embeddings.append(self.all_tracklets_dict[target_id.item()].embedding.detach())
        target_embeddings = torch.cat(target_embeddings)
        distractor_embeddings = []
        for distractor_id in distractor_ref_ids:
            distractor_embeddings.append(self.all_tracklets_dict[distractor_id.item()].embedding.detach())
        if len(distractor_embeddings):
            distractor_embeddings = torch.cat(distractor_embeddings)
        else:
            distractor_bools = None
            distractor_embeddings = None
        return target_bools, target_embeddings, distractor_bools, distractor_embeddings, distractor_ious

    def attention_training_match_ids(
        self,
        proposals,
        targets,
        min_overlap=0.5,
        min_distractor_overlap=0.2
    ):
        """
            Get proposals that match gt boxes with minimum overlap
            Only train attention on such positive proposals
            Then match proposals with other gt boxes to seek respective distractors
            min_distractor_overlap must be smaller than min_overlap
        """
        cat_matched_bools = []
        cat_matched_ids = []
        cat_matched_distractor_bools = []
        cat_matched_distractor_ids = []
        cat_distractor_ious = []
        for p, t in zip(proposals, targets):
            gt_boxes = t["boxes"].to(p[0].dtype)
            gt_ids = t["gt_ids"]

            # Match proposals to gt
            match_quality_matrix = box_iou(gt_boxes, p)
            top_vals, top_matches = match_quality_matrix.topk(k=2, dim=0)  # For each proposal

            # No.1 match should be corresponding target, if matched
            matched_bools = top_vals[0] > min_overlap
            matched_ids = gt_ids[top_matches[0]][matched_bools]
            cat_matched_bools.append(matched_bools)
            cat_matched_ids.append(matched_ids)

            # No.2 match should be corresponding distractor, if matched
            matched_distractor_bools = top_vals[1] > min_distractor_overlap
            matched_distractor_ids = gt_ids[top_matches[1]][matched_distractor_bools]
            cat_matched_distractor_bools.append(matched_distractor_bools)
            cat_matched_distractor_ids.append(matched_distractor_ids)
            cat_distractor_ious.append(top_vals[1])
        # Length of bools is original length
        cat_matched_bools = torch.cat(cat_matched_bools)
        cat_matched_distractor_bools = torch.cat(cat_matched_distractor_bools)
        # Length of ids is matched length, shorter than original length
        cat_matched_ids = torch.cat(cat_matched_ids)
        cat_matched_distractor_ids = torch.cat(cat_matched_distractor_ids)
        cat_distractor_ious = torch.cat(cat_distractor_ious)
        return cat_matched_bools, cat_matched_ids, cat_matched_distractor_bools, cat_matched_distractor_ids, cat_distractor_ious


class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None
