import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def indices_to_matches(cost_matrix, indices, threshold):
    """
        Generates three list of indices
    """
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= threshold)
    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))
    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, threshold):
    """
        Cost above threshold will not be assigned
        return: matches, unmatched_a, unmatched_b
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    cost_matrix[cost_matrix > threshold] = threshold + 1e-4
    indices = linear_sum_assignment(cost_matrix)  # In two arrays, like [[1,2,3,4,5], [3,4,2,5,1]]
    # Convert to pairs like [1, 3]
    indices_array = np.transpose(np.asarray(indices))
    return indices_to_matches(cost_matrix, indices_array, threshold)


def iou(altwh, bltwh):
    # Get coordinates of intersection
    xA = max(altwh[0], bltwh[0])
    yA = max(altwh[1], bltwh[1])
    xB = min(altwh[0] + altwh[2], bltwh[0] + bltwh[2])
    yB = min(altwh[1] + altwh[3], bltwh[1] + bltwh[3])

    intersection = max(0, xB - xA) * max(0, yB - yA)

    areaA = altwh[2] * altwh[3]
    areaB = bltwh[2] * bltwh[3]
    union = areaA + areaB - intersection

    return intersection / union


def reid_distance(tracklets, detections):
    cost_matrix = torch.zeros(len(tracklets), len(detections))
    if cost_matrix.shape[0] * cost_matrix.shape[1] == 0:
        return cost_matrix.numpy()

    for i, t in enumerate(tracklets):
        for j, det in enumerate(detections):
            cost_matrix[i, j] = 1 - F.cosine_similarity(t.avg_embedding, det.avg_embedding)
    return np.maximum(cost_matrix.detach().cpu().numpy(), 0.0)


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_giou(boxes1, boxes2):
    """
    Intersection-over-Union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Returns a Tensor[N, M] for all possible pairs
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # Get coordinates of potential common area
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    # Common area
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    # Union of the two
    union = area1[:, None] + area2 - inter

    # Get coordinates of smallest convex hull covering A and B
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt)  # [N,M,2]
    # Convex hull
    areaC = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    return inter / union - (areaC - union) / areaC
