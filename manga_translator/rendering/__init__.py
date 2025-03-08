import os
import cv2
import numpy as np
from typing import List
from shapely import affinity
from shapely.geometry import Polygon
from tqdm import tqdm
import math

# from .ballon_extractor import extract_ballon_region
from . import text_render
from .text_render_eng import render_textblock_list_eng
from ..utils import (
    BASE_PATH,
    TextBlock,
    color_difference,
    get_logger,
    rotate_polygons,
)

logger = get_logger('render')

def parse_font_paths(path: str, default: List[str] = None) -> List[str]:
    if path:
        parsed = path.split(',')
        parsed = list(filter(lambda p: os.path.isfile(p), parsed))
    else:
        parsed = default or []
    return parsed

def fg_bg_compare(fg, bg):
    fg_avg = np.mean(fg)
    if color_difference(fg, bg) < 30:
        bg = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
    return fg, bg

def resize_regions_to_font_size(img: np.ndarray, text_regions: List[TextBlock], font_size_fixed: int, font_size_offset: int, font_size_minimum: int):
    if font_size_minimum == -1:
        # Better balance for font size minimum to ensure text is always visible
        font_size_minimum = round((img.shape[0] + img.shape[1]) / 250)  # Increased for larger text
    logger.debug(f'font_size_minimum {font_size_minimum}')

    # Set a more balanced maximum font size
    font_size_maximum = round((img.shape[0] + img.shape[1]) / 50)  # Increased for larger text
    logger.debug(f'font_size_maximum {font_size_maximum}')
    
    # Ensure font_size_minimum is a valid integer and not too small
    try:
        font_size_minimum = int(font_size_minimum)
        font_size_minimum = max(14, font_size_minimum)  # Increased minimum for better readability
    except (TypeError, ValueError):
        font_size_minimum = 14  # Increased safe default
    
    # Ensure font_size_maximum is a valid integer
    try:
        font_size_maximum = int(font_size_maximum)
        font_size_maximum = max(32, font_size_maximum)  # Increased maximum
    except (TypeError, ValueError):
        font_size_maximum = 32  # Increased default
    
    # Ensure font_size_offset is a valid integer
    try:
        font_size_offset = int(font_size_offset) if font_size_offset is not None else 0
    except (TypeError, ValueError):
        font_size_offset = 0  # Safe default
    
    dst_points_list = []
    for region in text_regions:
        char_count_orig = len(region.text)
        char_count_trans = len(region.translation.strip())
        
        # Calculate aspect ratio to identify horizontal text bars
        width, height = region.unrotated_size[1], region.unrotated_size[0]
        is_horizontal_bar = width > height * 3  # Text is likely in a horizontal bar
        
        # Ensure region.font_size is a valid integer
        try:
            region.font_size = int(region.font_size)
            region.font_size = max(14, region.font_size)  # Increased minimum
        except (TypeError, ValueError):
            region.font_size = font_size_minimum  # Safe default
        
        # More balanced font size reduction for all text regions
        # Calculate area and estimate how many characters can fit
        area = width * height
        char_size_factor = 0.85  # Adjusted for better balance
        
        # Use this to set a maximum font size based on text length and available area, but with limits
        if char_count_trans > 0:
            # Estimate maximum font size that would allow all text to fit, with a reasonable minimum
            estimated_font_size = int(math.sqrt(area / (char_count_trans / char_size_factor)))
            estimated_font_size = max(14, estimated_font_size)  # Increased minimum
            # Cap the font size based on this estimate, but don't make it too small
            region.font_size = max(14, min(region.font_size, estimated_font_size))
            
        if char_count_trans > char_count_orig:
            # More characters were added, have to reduce fontsize to fit allotted area
            rescaled_font_size = region.font_size
            while True:
                rows = region.unrotated_size[0] // rescaled_font_size
                cols = region.unrotated_size[1] // rescaled_font_size
                
                # Additional constraint for horizontal bars to ensure readable text
                if is_horizontal_bar:
                    # For horizontal bars, ensure height can accommodate at least 1.5 lines
                    min_height_needed = int(rescaled_font_size * 1.5)
                    if region.unrotated_size[0] < min_height_needed:
                        rescaled_font_size = max(int(region.unrotated_size[0] // 1.5), font_size_minimum)
                
                if rows * cols >= char_count_trans or rescaled_font_size <= 14:
                    # Stop if we fit the text OR we've reached the minimum size
                    region.font_size = max(14, int(rescaled_font_size))
                    break
                rescaled_font_size -= 1
                if rescaled_font_size <= 0:
                    region.font_size = max(14, font_size_minimum)
                    break
        elif is_horizontal_bar and region.font_size > font_size_minimum:
            # For horizontal bars, ensure font size is appropriate for the height
            ideal_height = min(region.unrotated_size[0] * 0.7, font_size_maximum)
            if region.font_size > ideal_height:
                region.font_size = max(14, int(max(int(ideal_height), font_size_minimum)))

        # Infer the target fontsize
        target_font_size = region.font_size
        if font_size_fixed is not None:
            try:
                target_font_size = int(font_size_fixed)
                target_font_size = max(14, target_font_size)  # Increased minimum
            except (TypeError, ValueError):
                target_font_size = region.font_size  # Keep existing if conversion fails
        elif target_font_size < font_size_minimum:
            target_font_size = max(region.font_size, font_size_minimum)
        
        # Apply offset and ensure it doesn't exceed reasonable maximum for the image
        target_font_size += font_size_offset
        target_font_size = min(target_font_size, font_size_maximum)
        
        # More reasonable reduction for non-horizontal bars
        if not is_horizontal_bar:
            # Apply a more balanced scaling for normal text boxes
            target_font_size = max(14, int(target_font_size * 0.95))  # Less aggressive reduction
        
        # Ensure target_font_size is a valid integer and not too small
        target_font_size = max(14, int(target_font_size))
        
        logger.debug(f"Region font size: {target_font_size}, chars: {char_count_trans}, area: {area}, horizontal: {is_horizontal_bar}")

        # Rescale dst_points accordingly
        if target_font_size != region.font_size:
            target_scale = target_font_size / region.font_size
            dst_points = region.unrotated_min_rect[0]
            poly = Polygon(region.unrotated_min_rect[0])
            poly = affinity.scale(poly, xfact=target_scale, yfact=target_scale)
            dst_points = np.array(poly.exterior.coords[:4])
            dst_points = rotate_polygons(region.center, dst_points.reshape(1, -1), -region.angle).reshape(-1, 4, 2)

            # Clip to img width and height
            dst_points[..., 0] = dst_points[..., 0].clip(0, img.shape[1])
            dst_points[..., 1] = dst_points[..., 1].clip(0, img.shape[0])

            dst_points = dst_points.reshape((-1, 4, 2))
            region.font_size = int(target_font_size)
        else:
            dst_points = region.min_rect

        dst_points_list.append(dst_points)
    return dst_points_list

async def dispatch(
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_path: str = '',
    font_size_fixed: int = None,
    font_size_offset: int = 0,
    font_size_minimum: int = 0,
    hyphenate: bool = True,
    render_mask: np.ndarray = None,
    line_spacing: int = None,
    disable_font_border: bool = False
    ) -> np.ndarray:

    text_render.set_font(font_path)
    text_regions = list(filter(lambda region: region.translation, text_regions))

    # Resize regions that are too small
    dst_points_list = resize_regions_to_font_size(img, text_regions, font_size_fixed, font_size_offset, font_size_minimum)

    # TODO: Maybe remove intersections

    # Render text
    for region, dst_points in tqdm(zip(text_regions, dst_points_list), '[render]', total=len(text_regions)):
        if render_mask is not None:
            # set render_mask to 1 for the region that is inside dst_points
            cv2.fillConvexPoly(render_mask, dst_points.astype(np.int32), 1)
        img = render(img, region, dst_points, hyphenate, line_spacing, disable_font_border)
    return img

def render(
    img,
    region: TextBlock,
    dst_points,
    hyphenate,
    line_spacing,
    disable_font_border
):
    fg, bg = region.get_font_colors()
    fg, bg = fg_bg_compare(fg, bg)

    if disable_font_border :
        bg = None

    middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
    norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
    norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
    r_orig = np.mean(norm_h / norm_v)

    if region.horizontal:
        temp_box = text_render.put_text_horizontal(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_h[0]),
            round(norm_v[0]),
            region.alignment,
            region.direction == 'hr',
            fg,
            bg,
            region.target_lang,
            hyphenate,
            line_spacing,
        )
    else:
        temp_box = text_render.put_text_vertical(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_v[0]),
            region.alignment,
            fg,
            bg,
            line_spacing,
        )
    h, w, _ = temp_box.shape
    r_temp = w / h

    # Extend temporary box so that it has same ratio as original
    if r_temp > r_orig:
        h_ext = int(w / (2 * r_orig) - h / 2)
        box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)
        box[h_ext:h + h_ext, 0:w] = temp_box
    else:
        w_ext = int((h * r_orig - w) / 2)
        box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)
        box[0:h, w_ext:w_ext+w] = temp_box

    src_points = np.array([[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]]).astype(np.float32)
    #src_pts[:, 0] = np.clip(np.round(src_pts[:, 0]), 0, enlarged_w * 2)
    #src_pts[:, 1] = np.clip(np.round(src_pts[:, 1]), 0, enlarged_h * 2)

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    rgba_region = cv2.warpPerspective(box, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x, y, w, h = cv2.boundingRect(dst_points.astype(np.int32))
    canvas_region = rgba_region[y:y+h, x:x+w, :3]
    mask_region = rgba_region[y:y+h, x:x+w, 3:4].astype(np.float32) / 255.0
    img[y:y+h, x:x+w] = np.clip((img[y:y+h, x:x+w].astype(np.float32) * (1 - mask_region) + canvas_region.astype(np.float32) * mask_region), 0, 255).astype(np.uint8)
    return img

async def dispatch_eng_render(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: List[TextBlock], font_path: str = '', line_spacing: int = 0, disable_font_border: bool = False) -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = os.path.join(BASE_PATH, 'fonts/comic shanns 2.ttf')
    text_render.set_font(font_path)

    # Preprocess text regions to ensure better font sizing for horizontal bars
    for region in text_regions:
        # Ensure region.font_size starts as a valid integer
        try:
            region.font_size = int(region.font_size)
            region.font_size = max(14, region.font_size)  # Increased minimum
        except (TypeError, ValueError):
            region.font_size = 20  # Increased for better readability
        
        # Calculate aspect ratio
        width, height = region.unrotated_size[1], region.unrotated_size[0]
        is_horizontal_bar = width > height * 3  # Text is likely in a horizontal bar
        
        # Calculate area and character density with more balanced approach
        area = width * height
        char_count = len(region.translation.strip())
        
        if is_horizontal_bar:
            # For horizontal bars, use a font size that's more appropriate for the height
            # Use approximately 70% of the height as the font size to ensure readability
            try:
                optimal_font_size = int(height * 0.7)
                # But ensure it stays within reasonable bounds
                region.font_size = max(14, min(optimal_font_size, 36))
            except (TypeError, ValueError):
                region.font_size = 20  # Increased for better readability
            logger.debug(f"Adjusted font size for horizontal bar: {region.font_size}")
        else:
            # For normal text boxes, use more balanced font sizes
            region.font_size = min(region.font_size, 32)  # Increased maximum for normal text boxes
            
            # More balanced approach to character density
            if char_count > 0 and area > 0:
                # Estimate appropriate font size based on available area and text length
                # Higher char_count should result in smaller font, but with a minimum
                density_based_size = int(math.sqrt(area / (char_count * 0.9)))  # Less aggressive factor
                density_based_size = max(14, density_based_size)  # Increased minimum
                region.font_size = min(region.font_size, density_based_size)
                
                # Apply gentler reduction for crowded text boxes
                if char_count > width / 10:  # Less aggressive threshold
                    region.font_size = max(14, int(region.font_size * 0.95))  # Less aggressive reduction
                    
            logger.debug(f"Normal text box size: {region.font_size}, chars: {char_count}, area: {area}")

    # Ensure line_spacing is a valid int or float
    try:
        line_spacing = float(line_spacing) if line_spacing is not None else 0
    except (TypeError, ValueError):
        line_spacing = 0  # Safe default

    return render_textblock_list_eng(img_canvas, text_regions, line_spacing=line_spacing, 
                                     size_tol=1.2, original_img=original_img, 
                                     downscale_constraint=0.8,
                                     disable_font_border=disable_font_border)
