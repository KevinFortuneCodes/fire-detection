"""
FCOS (Fully-Convolutional One-Stage) Object Detector Implementation.

This module contains classes and functions for FCOS, a one-stage object detector.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision import models
from torchvision.models import feature_extraction
from torchvision.ops import sigmoid_focal_loss


def hello_fcos():
    print("Hello from fcos.py!")


class DetectorBackboneWithFPN(nn.Module):
    """
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32
    """

    def __init__(self, out_channels: int):
        super().__init__()

        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights for faster convergence.
        # Support both old (pretrained=True) and new (weights=...) torchvision versions
        try:
            # Try new API first (torchvision >= 0.13)
            _cnn = models.regnet_x_400mf(weights="IMAGENET1K_V2")
        except TypeError:
            # Fall back to old API (torchvision < 0.13)
            _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        # Initialize FPN layers
        self.fpn_params = nn.ModuleDict()

        # Lateral 1x1 conv layers to transform (c3, c4, c5) to same channels
        self.fpn_params["c3"] = nn.Conv2d(
            dummy_out["c3"].shape[1], self.out_channels, kernel_size=1
        )
        self.fpn_params["c4"] = nn.Conv2d(
            dummy_out["c4"].shape[1], self.out_channels, kernel_size=1
        )
        self.fpn_params["c5"] = nn.Conv2d(
            dummy_out["c5"].shape[1], self.out_channels, kernel_size=1
        )

        # Output 3x3 conv layers for p3, p4, p5
        # p3 uses (Conv, BN, ReLU, Conv) structure
        self.fpn_params["p3"] = nn.Sequential(
            nn.Conv2d(
                self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1
            ),
        )

        # p4 and p5 use single 3x3 conv
        self.fpn_params["p4"] = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1
        )
        self.fpn_params["p5"] = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1
        )

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):
        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}

        # Process FPN features
        c3 = self.fpn_params["c3"](backbone_feats["c3"])
        c4 = self.fpn_params["c4"](backbone_feats["c4"])
        c5 = self.fpn_params["c5"](backbone_feats["c5"])

        # Top-down pathway
        fpn_feats["p5"] = self.fpn_params["p5"](c5)

        p5_up = F.interpolate(
            fpn_feats["p5"], size=c4.shape[-2:], mode="nearest"
        )
        fpn_feats["p4"] = self.fpn_params["p4"](c4 + p5_up)

        p4_up = F.interpolate(
            fpn_feats["p4"], size=c3.shape[-2:], mode="nearest"
        )
        fpn_feats["p3"] = self.fpn_params["p3"](c3 + p4_up)

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates.
    """
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]
        B, C, H, W = feat_shape

        x_val = level_stride * (torch.arange(W, dtype=dtype, device=device) + 0.5)
        y_val = level_stride * (torch.arange(H, dtype=dtype, device=device) + 0.5)

        x_grid, y_grid = torch.meshgrid(y_val, x_val, indexing="ij")
        final_coordinate = torch.stack([x_grid, y_grid], dim=-1)
        final_coordinate = final_coordinate.reshape(H * W, 2)
        location_coords[level_name] = final_coordinate

    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
        scores: Tensor of shape (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores.
    """
    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    final_indices = []
    original_indices = torch.arange(boxes.size(0), device=boxes.device)

    while boxes.shape[0] > 0:
        max_idx = torch.argmax(scores)
        max_box = boxes[max_idx]
        final_indices.append(original_indices[max_idx].item())

        x1 = torch.max(max_box[0], boxes[:, 0])
        y1 = torch.max(max_box[1], boxes[:, 1])
        x2 = torch.min(max_box[2], boxes[:, 2])
        y2 = torch.min(max_box[3], boxes[:, 3])

        inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        box1_area = (max_box[2] - max_box[0]) * (max_box[3] - max_box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box1_area + boxes_area - inter_area
        iou = inter_area / union_area

        # Suppress boxes with high overlap
        keep_mask = iou <= iou_threshold
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        original_indices = original_indices[keep_mask]

    keep = torch.tensor(final_indices, dtype=torch.long)
    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]


class FCOSPredictionNetwork(nn.Module):
    """
    FCOS prediction network that accepts FPN feature maps from different levels
    and makes three predictions at every location: bounding boxes, class ID and
    centerness.
    """

    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):
        """
        Args:
            num_classes: Number of object classes for classification.
            in_channels: Number of channels in input feature maps.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
        """
        super().__init__()

        # Create stems for class and box
        stem_cls = []
        stem_box = []

        input_channel = in_channels
        for i in stem_channels:
            cls_conv = nn.Conv2d(input_channel, i, kernel_size=3, stride=1, padding=1)
            box_conv = nn.Conv2d(input_channel, i, kernel_size=3, stride=1, padding=1)

            nn.init.normal_(cls_conv.weight, mean=0, std=0.01)
            nn.init.constant_(cls_conv.bias, 0)
            nn.init.normal_(box_conv.weight, mean=0, std=0.01)
            nn.init.constant_(box_conv.bias, 0)

            stem_cls.append(cls_conv)
            stem_cls.append(nn.ReLU())
            stem_box.append(box_conv)
            stem_box.append(nn.ReLU())

            input_channel = i

        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)

        # Prediction layers
        self.pred_cls = nn.Conv2d(
            stem_channels[-1], num_classes, kernel_size=3, stride=1, padding=1
        )
        self.pred_box = nn.Conv2d(
            stem_channels[-1], 4, kernel_size=3, stride=1, padding=1
        )
        self.pred_ctr = nn.Conv2d(
            stem_channels[-1], 1, kernel_size=3, stride=1, padding=1
        )

        nn.init.normal_(self.pred_cls.weight, mean=0, std=0.01)
        nn.init.constant_(self.pred_cls.bias, 0)
        nn.init.normal_(self.pred_box.weight, mean=0, std=0.01)
        nn.init.constant_(self.pred_box.bias, 0)
        nn.init.normal_(self.pred_ctr.weight, mean=0, std=0.01)
        nn.init.constant_(self.pred_ctr.bias, 0)

        # Use a negative bias in `pred_cls` to improve training stability
        torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict the desired outputs at every location.

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}. Each
                tensor will have shape `(batch_size, fpn_channels, H, W)`.

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Classification logits: `(batch_size, H * W, num_classes)`.
            2. Box regression deltas: `(batch_size, H * W, 4)`
            3. Centerness logits:     `(batch_size, H * W, 1)`
        """
        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}

        for level_name in feats_per_fpn_level:
            B, C, H, W = feats_per_fpn_level[level_name].shape

            # Classification
            intermed_cls_out = self.stem_cls(feats_per_fpn_level[level_name])
            cls_pred = self.pred_cls(intermed_cls_out)
            cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(B, H * W, -1)
            class_logits[level_name] = cls_pred

            # Box regression
            intermed_box_out = self.stem_box(feats_per_fpn_level[level_name])
            box_pred = self.pred_box(intermed_box_out)
            box_pred = box_pred.permute(0, 2, 3, 1).reshape(B, H * W, 4)
            boxreg_deltas[level_name] = box_pred

            # Centerness
            intermed_center_out = self.stem_box(feats_per_fpn_level[level_name])
            center_pred = self.pred_ctr(intermed_center_out)
            center_pred = center_pred.permute(0, 2, 3, 1).reshape(B, H * W, 1)
            centerness_logits[level_name] = center_pred

        return [class_logits, boxreg_deltas, centerness_logits]


@torch.no_grad()
def fcos_match_locations_to_gt(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    gt_boxes: torch.Tensor,
) -> TensorDict:
    """
    Match centers of the locations of FPN feature with a set of GT bounding
    boxes of the input image.

    Args:
        locations_per_fpn_level: Centers at different levels of FPN (p3, p4, p5),
            that are already projected to absolute co-ordinates in input image
            dimension. Dictionary of three keys: (p3, p4, p5) giving tensors of
            shape `(H * W, 2)`.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
        gt_boxes: GT boxes of a single image, a batch of `(M, 5)` boxes with
            absolute co-ordinates and class ID `(x1, y1, x2, y2, C)`.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(N, 5)` GT boxes, one for each center.
    """
    matched_gt_boxes = {
        level_name: None for level_name in locations_per_fpn_level.keys()
    }
    
    # Handle empty gt_boxes (images with no annotations)
    if gt_boxes.shape[0] == 0:
        # Return all -1 boxes for all levels (indicating no matches)
        # Use the device from the first location tensor if gt_boxes is empty
        sample_centers = next(iter(locations_per_fpn_level.values()))
        device = sample_centers.device
        dtype = gt_boxes.dtype if gt_boxes.numel() > 0 else torch.float32
        
        for level_name, centers in locations_per_fpn_level.items():
            num_centers = centers.shape[0]
            matched_gt_boxes[level_name] = torch.full(
                (num_centers, 5), -1.0, dtype=dtype, device=device
            )
        return matched_gt_boxes

    for level_name, centers in locations_per_fpn_level.items():
        stride = strides_per_fpn_level[level_name]
        x, y = centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes[:, :4].unsqueeze(dim=0).unbind(dim=2)

        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)
        pairwise_dist = pairwise_dist.permute(1, 0, 2)

        # Anchor point must be inside GT
        match_matrix = pairwise_dist.min(dim=2).values > 0

        # Multilevel anchor matching: each anchor is only responsible for certain scale range
        pairwise_dist = pairwise_dist.max(dim=2).values
        lower_bound = stride * 4 if level_name != "p3" else 0
        upper_bound = stride * 8 if level_name != "p5" else float("inf")

        match_matrix &= (pairwise_dist > lower_bound) & (pairwise_dist < upper_bound)

        # Match the GT box with minimum area, if there are multiple GT matches
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        match_matrix = match_matrix.to(torch.float32)
        match_matrix *= 1e8 - gt_areas[:, None]

        match_quality, matched_idxs = match_matrix.max(dim=0)
        matched_idxs[match_quality < 1e-5] = -1

        matched_boxes_this_level = gt_boxes[matched_idxs.clip(min=0)]
        matched_boxes_this_level[matched_idxs < 0, :] = -1

        matched_gt_boxes[level_name] = matched_boxes_this_level

    return matched_gt_boxes


def fcos_get_deltas_from_locations(
    locations: torch.Tensor, gt_boxes: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Compute distances from feature locations to GT box edges.

    Args:
        locations: Tensor of shape `(N, 2)` giving `(xc, yc)` feature locations.
        gt_boxes: Tensor of shape `(N, 4 or 5)` giving GT boxes.
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            Tensor of shape `(N, 4)` giving deltas (left, top, right, bottom)
            from the locations to GT box edges, normalized by FPN stride.
    """
    xc, yc = locations[:, 0], locations[:, 1]
    x1, y1, x2, y2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]

    l = (xc - x1) / stride
    t = (yc - y1) / stride
    r = (x2 - xc) / stride
    b = (y2 - yc) / stride

    deltas = torch.stack([l, t, r, b], dim=1)
    deltas[(gt_boxes == -1).all(dim=1)] = -1

    return deltas


def fcos_apply_deltas_to_locations(
    deltas: torch.Tensor, locations: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Apply edge deltas to feature locations to get bounding box coordinates.

    Args:
        deltas: Tensor of shape `(N, 4)` giving edge deltas to apply to locations.
        locations: Locations to apply deltas on. shape: `(N, 2)`
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            Same shape as deltas, giving co-ordinates of the resulting boxes
            `(x1, y1, x2, y2)`, absolute in image dimensions.
    """
    deltas = deltas.clamp(min=0)  # Clip negative deltas to zero

    xc, yc = locations[:, 0], locations[:, 1]
    l, t, r, b = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    x1 = xc - l * stride
    y1 = yc - t * stride
    x2 = r * stride + xc
    y2 = b * stride + yc

    output_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return output_boxes


def fcos_make_centerness_targets(deltas: torch.Tensor):
    """
    Given LTRB deltas of GT boxes, compute GT targets for supervising the
    centerness regression predictor.

    Args:
        deltas: Tensor of shape `(N, 4)` giving LTRB deltas for GT boxes.

    Returns:
        torch.Tensor
            Tensor of shape `(N, )` giving centerness regression targets.
    """
    l, t, r, b = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    centerness = ((torch.min(l, r) * torch.min(t, b)) / (torch.max(l, r) * torch.max(t, b))) ** 0.5
    centerness[(deltas == -1).all(dim=1)] = -1

    return centerness


class FCOS(nn.Module):
    """
    FCOS: Fully-Convolutional One-Stage Detector

    This class puts together everything for FCOS. It contains a backbone with FPN,
    and prediction layers (head). It computes loss during training and predicts
    boxes during inference.
    """

    def __init__(
        self, num_classes: int, fpn_channels: int, stem_channels: List[int]
    ):
        super().__init__()

        self.num_classes = num_classes
        self.backbone = DetectorBackboneWithFPN(fpn_channels)
        self.pred_net = FCOSPredictionNetwork(
            self.num_classes, fpn_channels, stem_channels
        )

        # Averaging factor for training loss; EMA of foreground locations.
        self._normalizer = 150  # per image

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        Args:
            images: Batch of images, tensors of shape `(B, C, H, W)`.
            gt_boxes: Batch of training boxes, tensors of shape `(B, N, 5)`.
                `gt_boxes[i, j] = (x1, y1, x2, y2, C)` gives information about
                the `j`th object in `images[i]`. Not provided during inference.
            test_score_thresh: During inference, discard predictions with a
                confidence score less than this value. Ignored during training.
            test_nms_thresh: IoU threshold for NMS during inference. Ignored
                during training.

        Returns:
            Losses during training and predictions during inference.
        """
        # Process the image through backbone, FPN, and prediction head
        fpn_feats = self.backbone.forward(images)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.pred_net.forward(
            fpn_feats
        )

        # Get absolute co-ordinates `(xc, yc)` for every location in FPN levels
        fpn_feats_shapes = {
            level_name: feat.shape for level_name, feat in fpn_feats.items()
        }
        locations_per_fpn_level = get_fpn_location_coords(
            fpn_feats_shapes,
            self.backbone.fpn_strides,
            dtype=images.dtype,
            device=str(images.device),
        )

        # Only go to inference if we're not training AND gt_boxes is explicitly None (not provided)
        # IMPORTANT: Even if gt_boxes is an empty tensor (no annotations), we still want to compute loss
        # during validation. Only go to inference if gt_boxes is None (not passed at all).
        # This ensures validation (eval mode with gt_boxes, even if empty) computes loss, not inference
        # 
        # Safety: Check if gt_boxes is a tensor (even empty) - if so, we're in training/validation mode
        # Only go to inference if gt_boxes is explicitly None AND we're not training
        is_inference_mode = (
            not self.training and 
            gt_boxes is None  # Must be explicitly None, not an empty tensor
        )
        
        if is_inference_mode:
            # During inference, just go to this method and skip rest of the forward pass
            # Convert to float explicitly to ensure type safety
            score_thresh = float(test_score_thresh if test_score_thresh is not None else 0.3)
            nms_thresh = float(test_nms_thresh if test_nms_thresh is not None else 0.5)
            return self.inference(
                images,
                locations_per_fpn_level,
                pred_cls_logits,
                pred_boxreg_deltas,
                pred_ctr_logits,
                test_score_thresh=score_thresh,
                test_nms_thresh=nms_thresh,
            )

        # Assign ground-truth boxes to feature locations
        # Ensure gt_boxes is not None before iterating
        if gt_boxes is None:
            raise ValueError("gt_boxes cannot be None when computing loss. Use inference mode for prediction.")
        
        matched_gt_boxes = []
        for gt_box in gt_boxes:
            matched_gt_boxes_intermed = fcos_match_locations_to_gt(
                locations_per_fpn_level, self.backbone.fpn_strides, gt_box
            )
            matched_gt_boxes.append(matched_gt_boxes_intermed)

        # Calculate GT deltas for these matched boxes
        matched_gt_deltas = []
        for match_box in matched_gt_boxes:
            matched_gt_deltas_intermed = {}
            for level_name in match_box:
                input_boxes = match_box[level_name]
                input_locations = locations_per_fpn_level[level_name]
                deltas = fcos_get_deltas_from_locations(
                    input_locations, input_boxes, stride=self.backbone.fpn_strides[level_name]
                )
                matched_gt_deltas_intermed[level_name] = deltas
            matched_gt_deltas.append(matched_gt_deltas_intermed)

        # Collate lists of dictionaries, to dictionaries of batched tensors
        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)

        # Combine predictions and GT from across all FPN levels
        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)

        # Perform EMA update of normalizer by number of positive locations
        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / images.shape[0]
        self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image

        # Calculate losses per location
        B, L, C = pred_cls_logits.shape
        gt_classes = matched_gt_boxes[:, :, 4].long()

        gt_classes_one_hot = F.one_hot(
            torch.clamp(gt_classes, min=0), num_classes=self.num_classes
        ).to(dtype=pred_cls_logits.dtype)
        gt_classes_one_hot[gt_classes == -1] = 0

        loss_cls = sigmoid_focal_loss(
            inputs=pred_cls_logits,
            targets=gt_classes_one_hot,
            reduction="none",
        )

        # Box regression loss
        pred_boxreg_deltas_flat = pred_boxreg_deltas.view(-1, 4)
        matched_gt_deltas_flat = matched_gt_deltas.view(-1, 4)
        loss_box = 0.25 * F.l1_loss(
            pred_boxreg_deltas_flat,
            matched_gt_deltas_flat,
            reduction="none",
        )
        loss_box[matched_gt_deltas_flat < 0] *= 0
        loss_box = loss_box.view(B, L, 4)

        # Centerness loss
        centerness = torch.stack(
            [fcos_make_centerness_targets(i) for i in matched_gt_deltas], dim=0
        )
        pred_ctr_logits_flat = pred_ctr_logits.view(-1)
        centerness_flat = centerness.view(-1)
        loss_ctr = F.binary_cross_entropy_with_logits(
            pred_ctr_logits_flat, centerness_flat, reduction="none"
        )
        loss_ctr[centerness_flat < 0] *= 0
        loss_ctr = loss_ctr.view(B, L)

        # Sum all locations and average by the EMA of foreground locations
        return {
            "loss_cls": loss_cls.sum() / (self._normalizer * images.shape[0]),
            "loss_box": loss_box.sum() / (self._normalizer * images.shape[0]),
            "loss_ctr": loss_ctr.sum() / (self._normalizer * images.shape[0]),
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        Run inference on a single input image (batch size = 1).

        Args:
            test_score_thresh: Confidence score threshold. Defaults to 0.3 if None.
            test_nms_thresh: IoU threshold for NMS. Defaults to 0.5 if None.

        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes.
                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels).
                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions.
        """
        # Robust safety check: ensure thresholds are always float values, never None
        # Convert to float explicitly to handle any edge cases and create local immutable copies
        # Use try-except to ensure we always have valid values, even if conversion fails
        try:
            if test_score_thresh is None:
                score_threshold = 0.3
            elif isinstance(test_score_thresh, torch.Tensor):
                score_threshold = float(test_score_thresh.item())
            else:
                score_threshold = float(test_score_thresh)
        except (ValueError, TypeError, AttributeError) as e:
            # Fallback to default if conversion fails
            score_threshold = 0.3
        
        try:
            if test_nms_thresh is None:
                nms_threshold = 0.5
            elif isinstance(test_nms_thresh, torch.Tensor):
                nms_threshold = float(test_nms_thresh.item())
            else:
                nms_threshold = float(test_nms_thresh)
        except (ValueError, TypeError, AttributeError) as e:
            # Fallback to default if conversion fails
            nms_threshold = 0.5
        
        # Final validation: ensure we have valid float values
        # If somehow we still have None or invalid values, use defaults
        if score_threshold is None or not isinstance(score_threshold, (int, float)):
            score_threshold = 0.3
        if nms_threshold is None or not isinstance(nms_threshold, (int, float)):
            nms_threshold = 0.5
        
        # Convert to float one more time to ensure type consistency
        score_threshold = float(score_threshold)
        nms_threshold = float(nms_threshold)
        
        # Final assertion to ensure we never have None values
        assert score_threshold is not None, "score_threshold must not be None"
        assert nms_threshold is not None, "nms_threshold must not be None"
        assert isinstance(score_threshold, float), f"score_threshold must be float, got {type(score_threshold)}"
        assert isinstance(nms_threshold, float), f"nms_threshold must be float, got {type(nms_threshold)}"
        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in locations_per_fpn_level.keys():
            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]

            # Compute geometric mean of class probability and centerness
            level_pred_scores = torch.sqrt(
                level_cls_logits.sigmoid() * level_ctr_logits.sigmoid()
            )

            # Get the most confidently predicted class and its score for every box
            num_classes = level_pred_scores.shape[1]
            level_pred_scores_flat = level_pred_scores.reshape(-1)
            level_pred_classes_flat = torch.arange(
                num_classes, device=level_cls_logits.device
            ).repeat(level_cls_logits.shape[0])

            level_deltas_expanded = (
                level_deltas.unsqueeze(1).expand(-1, num_classes, -1).reshape(-1, 4)
            )
            level_locations_expanded = (
                level_locations.unsqueeze(1)
                .expand(-1, num_classes, -1)
                .reshape(-1, 2)
            )

            # Only retain predictions that have a confidence score higher than threshold
            # Use the local immutable score_threshold variable (already validated)
            mask = level_pred_scores_flat > score_threshold
            level_pred_scores_flat = level_pred_scores_flat[mask]
            level_pred_classes_flat = level_pred_classes_flat[mask]
            level_deltas_expanded = level_deltas_expanded[mask]
            level_locations_expanded = level_locations_expanded[mask]

            # Obtain predicted boxes using predicted deltas and locations
            stride = self.backbone.fpn_strides[level_name]
            level_pred_boxes = fcos_apply_deltas_to_locations(
                level_deltas_expanded, level_locations_expanded, stride
            )

            # Clip XYXY box-coordinates that go beyond the height and width of input image
            height, width = images.shape[2], images.shape[3]
            level_pred_boxes[:, 0] = level_pred_boxes[:, 0].clamp(min=0)
            level_pred_boxes[:, 1] = level_pred_boxes[:, 1].clamp(min=0)
            level_pred_boxes[:, 2] = level_pred_boxes[:, 2].clamp(max=width)
            level_pred_boxes[:, 3] = level_pred_boxes[:, 3].clamp(max=height)

            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes_flat)
            pred_scores_all_levels.append(level_pred_scores_flat)

        # Combine predictions from all levels and perform NMS
        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        # Use the local immutable nms_threshold variable (already validated)
        keep = class_spec_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=nms_threshold,  # Use validated local variable
        )

        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]

        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )

