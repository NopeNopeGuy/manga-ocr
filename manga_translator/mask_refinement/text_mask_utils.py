from typing import Tuple, List
import numpy as np
import cv2
import math

from tqdm import tqdm
from shapely.geometry import Polygon
# from sklearn.mixture import BayesianGaussianMixture
# from functools import reduce
# from collections import defaultdict
# from scipy.optimize import linear_sum_assignment

from ..utils import Quadrilateral, image_resize

COLOR_RANGE_SIGMA = 1.5 # how many stddev away is considered the same color

def save_rgb(fn, img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        cv2.imwrite(fn, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(fn, img)

def area_overlap(x1, y1, w1, h1, x2, y2, w2, h2):  # returns None if rectangles don't intersect
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return x_overlap * y_overlap

def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

def rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist(x1, y1b, x2b, y2)
    elif left and bottom:
        return dist(x1, y1, x2b, y2b)
    elif bottom and right:
        return dist(x1b, y1, x2, y2b)
    elif right and top:
        return dist(x1b, y1b, x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:             # rectangles intersect
        return 0
    
def extend_rect(x, y, w, h, max_x, max_y, extend_size):
    x1 = max(x - extend_size, 0)
    y1 = max(y - extend_size, 0)
    w1 = min(w + extend_size * 2, max_x - x1 - 1)
    h1 = min(h + extend_size * 2, max_y - y1 - 1)
    return x1, y1, w1, h1

def complete_mask_fill(text_lines: List[Tuple[int, int, int, int]]):
    for (x, y, w, h) in text_lines:
        final_mask = cv2.rectangle(final_mask, (x, y), (x + w, y + h), (255), -1)
    return final_mask

from pydensecrf.utils import compute_unary, unary_from_softmax
import pydensecrf.densecrf as dcrf

def refine_mask(rgbimg, rawmask):
    if len(rawmask.shape) == 2:
        rawmask = rawmask[:, :, None]
    mask_softmax = np.concatenate([cv2.bitwise_not(rawmask)[:, :, None], rawmask], axis=2)
    mask_softmax = mask_softmax.astype(np.float32) / 255.0
    n_classes = 2
    feat_first = mask_softmax.transpose((2, 0, 1)).reshape((n_classes,-1))
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(rgbimg.shape[1], rgbimg.shape[0], n_classes)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=1, compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NO_NORMALIZATION)

    d.addPairwiseBilateral(sxy=20, srgb=10, rgbim=rgbimg,
                        compat=15,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NO_NORMALIZATION)
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((rgbimg.shape[0], rgbimg.shape[1]))
    crf_mask = np.array(res * 255, dtype=np.uint8)
    return crf_mask

def complete_mask(img: np.ndarray, mask: np.ndarray, textlines: List[Quadrilateral], keep_threshold = 5e-3, dilation_offset = 0, kernel_size=3):
    bboxes = [txtln.aabb.xywh for txtln in textlines]
    polys = [Polygon(txtln.pts) for txtln in textlines]
    
    # Create an initial mask highlighting all text areas to avoid filtering out vertical lines
    initial_mask = np.zeros_like(mask)
    for (x, y, w, h) in bboxes:
        cv2.rectangle(initial_mask, (x, y), (x + w, y + h), (255), 1)
    
    # Draw boundaries around text boxes in the mask
    for (x, y, w, h) in bboxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), (0), 1)
    
    # Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # Pre-process to find vertical and horizontal lines that might be important
    vertical_lines_mask = np.zeros_like(mask)
    horizontal_lines_mask = np.zeros_like(mask)
    
    # More aggressive line detection
    for label in range(1, num_labels):
        x1 = stats[label, cv2.CC_STAT_LEFT]
        y1 = stats[label, cv2.CC_STAT_TOP]
        w1 = stats[label, cv2.CC_STAT_WIDTH]
        h1 = stats[label, cv2.CC_STAT_HEIGHT]
        area1 = stats[label, cv2.CC_STAT_AREA]
        
        # Detect both vertical and horizontal lines with more relaxed criteria
        is_line = area1 >= 5 and (w1 * h1) / area1 < 2.0  # More solid shapes have ratio closer to 1
        
        # Check if this component is likely a vertical line
        if is_line and (h1 > w1 * 2.5) and (w1 < 20):
            vertical_lines_mask[labels == label] = 255
            
        # Check if this component is likely a horizontal line
        if is_line and (w1 > h1 * 2.5) and (h1 < 20):
            horizontal_lines_mask[labels == label] = 255
    
    # Combine all line masks
    lines_mask = cv2.bitwise_or(vertical_lines_mask, horizontal_lines_mask)
    
    # Dilate the lines mask slightly to connect fragmented lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lines_mask = cv2.dilate(lines_mask, kernel, iterations=1)

    M = len(textlines)
    textline_ccs = [np.zeros_like(mask) for _ in range(M)]
    iinfo = np.iinfo(labels.dtype)
    textline_rects = np.full(shape = (M, 4), fill_value = [iinfo.max, iinfo.max, iinfo.min, iinfo.min], dtype = labels.dtype)
    ratio_mat = np.zeros(shape = (num_labels, M), dtype = np.float32)
    dist_mat = np.zeros(shape = (num_labels, M), dtype = np.float32)
    valid = False
    
    for label in range(1, num_labels):
        # Skip very small areas
        if stats[label, cv2.CC_STAT_AREA] <= 5:
            continue

        x1 = stats[label, cv2.CC_STAT_LEFT]
        y1 = stats[label, cv2.CC_STAT_TOP]
        w1 = stats[label, cv2.CC_STAT_WIDTH]
        h1 = stats[label, cv2.CC_STAT_HEIGHT]
        area1 = stats[label, cv2.CC_STAT_AREA]
        cc_pts = np.array([[x1, y1], [x1 + w1, y1], [x1 + w1, y1 + h1], [x1, y1 + h1]])
        cc_poly = Polygon(cc_pts)
        
        # Check if this is likely a line (vertical or horizontal)
        is_vertical_line = (h1 > w1 * 2.5) and (w1 < 20)
        is_horizontal_line = (w1 > h1 * 2.5) and (h1 < 20)
        is_line = is_vertical_line or is_horizontal_line

        for tl_idx in range(M):
            area2 = polys[tl_idx].area
            overlapping_area = polys[tl_idx].intersection(cc_poly).area
            ratio_mat[label, tl_idx] = overlapping_area / min(area1, area2)
            dist_mat[label, tl_idx] = polys[tl_idx].distance(cc_poly.centroid)

        avg = np.argmax(ratio_mat[label])
        area2 = polys[avg].area
        
        # More lenient criteria for lines
        if is_line:
            # Lower threshold for lines to be included
            if ratio_mat[label, avg] <= keep_threshold * 0.3:
                avg = np.argmin(dist_mat[label])
                area2 = polys[avg].area
                unit = max(min([textlines[avg].font_size, max(w1, h1)]), 10)
                
                # More forgiving distance threshold for lines
                if dist_mat[label, avg] >= 1.2 * unit:
                    continue
        else:
            # Standard criteria for non-line elements
            if area1 >= area2:
                continue
            if ratio_mat[label, avg] <= keep_threshold:
                avg = np.argmin(dist_mat[label])
                area2 = polys[avg].area
                unit = max(min([textlines[avg].font_size, w1, h1]), 10)
                if dist_mat[label, avg] >= 0.5 * unit:
                    continue

        textline_ccs[avg][y1:y1+h1, x1:x1+w1][labels[y1:y1+h1, x1:x1+w1] == label] = 255
        textline_rects[avg, 0] = min(textline_rects[avg, 0], x1)
        textline_rects[avg, 1] = min(textline_rects[avg, 1], y1)
        textline_rects[avg, 2] = max(textline_rects[avg, 2], x1 + w1)
        textline_rects[avg, 3] = max(textline_rects[avg, 3], y1 + h1)
        valid = True

    # If no valid textlines found but we have lines, still process them
    if not valid and np.sum(lines_mask) > 0:
        return lines_mask
    elif not valid:
        return None
    
    # tblr to xywh
    textline_rects[:, 2] -= textline_rects[:, 0]
    textline_rects[:, 3] -= textline_rects[:, 1]
    
    final_mask = np.zeros_like(mask)
    img = cv2.bilateralFilter(img, 17, 80, 80)
    
    # Add detected lines to the final mask
    final_mask = cv2.bitwise_or(final_mask, lines_mask)
    
    for i, cc in enumerate(tqdm(textline_ccs, '[mask]')):
        x1, y1, w1, h1 = textline_rects[i]
        text_size = min(w1, h1, textlines[i].font_size)
        x1, y1, w1, h1 = extend_rect(x1, y1, w1, h1, img.shape[1], img.shape[0], int(text_size * 0.1))
        
        dilate_size = max((int((text_size + dilation_offset) * 0.3) // 2) * 2 + 1, 3)
        
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        cc_region = np.ascontiguousarray(cc[y1: y1 + h1, x1: x1 + w1])
        if cc_region.size == 0:
            continue
        
        img_region = np.ascontiguousarray(img[y1: y1 + h1, x1: x1 + w1])
        cc_region = refine_mask(img_region, cc_region)
        
        cc[y1: y1 + h1, x1: x1 + w1] = cc_region
        
        x2, y2, w2, h2 = extend_rect(x1, y1, w1, h1, img.shape[1], img.shape[0], -(-dilate_size // 2))
        cc[y2:y2+h2, x2:x2+w2] = cv2.dilate(cc[y2:y2+h2, x2:x2+w2], kern)
        final_mask[y2:y2+h2, x2:x2+w2] = cv2.bitwise_or(final_mask[y2:y2+h2, x2:x2+w2], cc[y2:y2+h2, x2:x2+w2])
    
    # Additional processing to connect any discontinuous line segments
    # Use a slightly larger kernel for final dilation to better connect elements
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size + 2, kernel_size + 2))
    dilated_mask = cv2.dilate(final_mask, kern)
    
    # Find gaps in vertical lines and fill them
    line_gaps = np.zeros_like(final_mask)
    for y in range(1, final_mask.shape[0]-1):
        for x in range(final_mask.shape[1]):
            # If this pixel is empty but pixels above and below are filled
            if final_mask[y, x] == 0 and final_mask[y-1, x] > 0 and final_mask[y+1, x] > 0:
                line_gaps[y, x] = 255
                
    # Add the filled gaps to our final mask
    final_mask = cv2.bitwise_or(dilated_mask, line_gaps)
    
    return final_mask

def unsharp(image):
    gaussian_3 = cv2.GaussianBlur(image, (3, 3), 2.0)
    return cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
