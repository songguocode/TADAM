import cv2
import numpy as np
from torchvision.transforms.functional import to_pil_image


def tensor_to_cv2(tensor_image):
    # To PIL image first
    pil_image = to_pil_image(tensor_image)
    opencv_image = np.array(pil_image)
    # Convert RGB to BGR
    opencv_image = opencv_image[:, :, ::-1].copy()
    return opencv_image


def cmc_align(last_frame, curr_frame):
    """
        Camera movement compensation
        Returns warp matrix for position alignment
    """
    last_gray = cv2.cvtColor(np.array(last_frame), cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(np.array(curr_frame), cv2.COLOR_RGB2GRAY)
    warp_mode = cv2.MOTION_EUCLIDEAN
    # warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 100
    termination_eps = 0.00001
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    (cc, warp_matrix) = cv2.findTransformECC(last_gray, curr_gray, warp_matrix, warp_mode, criteria, None, 5)
    return warp_matrix
