import itertools
import math
from typing import Callable, List, Set, Optional, Tuple, Union
from collections import defaultdict, Counter
import os
import shutil
import cv2
from PIL import Image
import numpy as np
import einops
import networkx as nx
from shapely.geometry import Polygon

import torch

# Replace MangaOcr import with our own implementation
from .manga_ocr_inference import MangaOCR

from .common import OfflineOCR
from .model_48px import OCR
from ..config import OcrConfig
from ..textline_merge import split_text_region
from ..utils import TextBlock, Quadrilateral, quadrilateral_can_merge_region, chunks
from ..utils.generic import AvgMeter

async def merge_bboxes(bboxes: List[Quadrilateral], width: int, height: int) -> Tuple[List[Quadrilateral], int]:
    if not bboxes:
        return [], []
    
    # Fast path for single bbox
    if len(bboxes) == 1:
        return bboxes, [[0]]
    
    # Create spatial grid for faster neighbor finding
    # This reduces O(n²) comparison to mostly local comparisons
    cell_size = min(width, height) / 20  # Adjust this value based on typical text size
    grid = defaultdict(list)
    
    # Map each bbox to grid cells based on its bounds
    for i, box in enumerate(bboxes):
        # Get expanded AABB for the box to ensure we catch nearby boxes
        x_min, y_min = np.min(box.pts, axis=0)
        x_max, y_max = np.max(box.pts, axis=0)
        
        # Calculate which grid cells this box overlaps
        min_cell_x = max(0, int((x_min - cell_size) / cell_size))
        max_cell_x = min(int(width / cell_size), int((x_max + cell_size) / cell_size) + 1)
        min_cell_y = max(0, int((y_min - cell_size) / cell_size))
        max_cell_y = min(int(height / cell_size), int((y_max + cell_size) / cell_size) + 1)
        
        # Add box index to all relevant grid cells
        for cell_x in range(min_cell_x, max_cell_x):
            for cell_y in range(min_cell_y, max_cell_y):
                grid[(cell_x, cell_y)].append(i)
    
    # Build graph with spatial optimization
    G = nx.Graph()
    for i, box in enumerate(bboxes):
        G.add_node(i, box=box)
    
    # Use grid to find potential neighbors
    processed_pairs = set()
    for cell, indices in grid.items():
        # Check all pairs within this cell
        for i, u in enumerate(indices):
            ubox = bboxes[u]
            # Check all other boxes in this cell
            for v in indices[i+1:]:
                if (u, v) in processed_pairs:
                    continue
                processed_pairs.add((u, v))
                processed_pairs.add((v, u))
                
                vbox = bboxes[v]
                # First do a fast check on font size ratio before the more expensive check
                if max(ubox.font_size, vbox.font_size) / min(ubox.font_size, vbox.font_size) <= 2:
                    if quadrilateral_can_merge_region(ubox, vbox, aspect_ratio_tol=1.3, font_size_ratio_tol=2,
                                                    char_gap_tolerance=0.8, char_gap_tolerance2=2.5):
                        G.add_edge(u, v)

    # Step 2: Postprocess - further split each region
    region_indices: List[Set[int]] = []
    for node_set in nx.algorithms.components.connected_components(G):
         region_indices.extend(split_text_region(bboxes, node_set, width, height))

    # Step 3: Return regions
    merge_box = []
    merge_idx = []
    for node_set in region_indices:
        nodes = list(node_set)
        txtlns: List[Quadrilateral] = np.array(bboxes)[nodes]

        # Majority vote for direction
        dirs = [box.direction for box in txtlns]
        majority_dir_top_2 = Counter(dirs).most_common(2)
        
        # Optimize the majority direction calculation
        if len(majority_dir_top_2) == 1:
            majority_dir = majority_dir_top_2[0][0]
        elif majority_dir_top_2[0][1] == majority_dir_top_2[1][1]:  # if top 2 have the same counts
            # Find the box with the maximum aspect ratio as it's likely more representative
            max_aspect_ratio = -100
            majority_dir = None
            for box in txtlns:
                aspect = max(box.aspect_ratio, 1.0 / box.aspect_ratio)
                if aspect > max_aspect_ratio:
                    max_aspect_ratio = aspect
                    majority_dir = box.direction
        else:
            majority_dir = majority_dir_top_2[0][0]

        # Sort textlines
        if majority_dir == 'h':
            nodes = sorted(nodes, key=lambda x: bboxes[x].centroid[1])
        elif majority_dir == 'v':
            nodes = sorted(nodes, key=lambda x: -bboxes[x].centroid[0])
        txtlns = np.array(bboxes)[nodes]
        
        # Yield overall bbox and sorted indices
        merge_box.append(txtlns)
        merge_idx.append(nodes)

    return_box = []
    # Process boxes and merge them
    for bbox in merge_box:
        if len(bbox) == 1:
            return_box.append(bbox[0])
        else:
            # Calculate average probability
            prob = sum(q.prob for q in bbox) / len(bbox)
            
            # Merge boxes efficiently
            base_box = bbox[0]
            if len(bbox) > 1:
                # Create a single polygon from all points to find minimum rectangle
                all_pts = []
                for box in bbox:
                    all_pts.extend(box.pts)
                min_rect = np.array(Polygon(all_pts).minimum_rotated_rectangle.exterior.coords[:4])
                base_box = Quadrilateral(min_rect, '', prob)
            
            return_box.append(base_box)
    
    return return_box, merge_idx

class ModelMangaOCR(OfflineOCR):
    _MODEL_MAPPING = {
        'model': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr_ar_48px.ckpt',
            'hash': '29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d',
        },
        'dict': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/alphabet-all-v7.txt',
            'hash': 'f5722368146aa0fbcc9f4726866e4efc3203318ebb66c811d8cbbe915576538a',
        },
        'onnx_model': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/quantized_model.onnx',  # Update with correct URL
            'hash': '',  # Update with correct hash
        },
    }

    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        if os.path.exists('ocr_ar_48px.ckpt'):
            shutil.move('ocr_ar_48px.ckpt', self._get_file_path('ocr_ar_48px.ckpt'))
        if os.path.exists('alphabet-all-v7.txt'):
            shutil.move('alphabet-all-v7.txt', self._get_file_path('alphabet-all-v7.txt'))
        if os.path.exists('quantized_model.onnx'):
            shutil.move('quantized_model.onnx', self._get_file_path('quantized_model.onnx'))
        super().__init__(*args, **kwargs)

    async def _load(self, device: str):
        with open(self._get_file_path('alphabet-all-v7.txt'), 'r', encoding = 'utf-8') as fp:
            dictionary = [s[:-1] for s in fp.readlines()]

        # Keep the original OCR model for backward compatibility
        self.model = OCR(dictionary, 768)
        sd = torch.load(self._get_file_path('ocr_ar_48px.ckpt'))
        self.model.load_state_dict(sd)
        self.model.eval()
        
        # Initialize the ONNX model instead of the original MangaOcr
        onnx_model_path = self._get_file_path('quantized_model.onnx')
        vocab_path = self._get_file_path('vocab.txt')
        self.mocr = MangaOCR(onnx_model_path, vocab_path)
        
        self.device = device
        if (device == 'cuda' or device == 'mps'):
            self.use_gpu = True
        else:
            self.use_gpu = False
        if self.use_gpu:
            self.model = self.model.to(device)

    async def _unload(self):
        del self.model
        del self.mocr

    async def _infer(self, image: np.ndarray, textlines: List[Quadrilateral], config: OcrConfig, verbose: bool = False, ignore_bubble: int = 0) -> List[TextBlock]:
        text_height = 48
        max_chunk_size = 16

        quadrilaterals = list(self._generate_text_direction(textlines))
        region_imgs = [q.get_transformed_region(image, d, text_height) for q, d in quadrilaterals]

        perm = range(len(region_imgs))
        is_quadrilaterals = False
        if len(quadrilaterals) > 0 and isinstance(quadrilaterals[0][0], Quadrilateral):
            perm = sorted(range(len(region_imgs)), key = lambda x: region_imgs[x].shape[1])
            is_quadrilaterals = True

        texts = {}
        if config.use_mocr_merge:
            merged_textlines, merged_idx = await merge_bboxes(textlines, image.shape[1], image.shape[0])
            merged_quadrilaterals = list(self._generate_text_direction(merged_textlines))
        else:
            merged_idx = [[i] for i in range(len(region_imgs))]
            merged_quadrilaterals = quadrilaterals
        merged_region_imgs = []
        for q, d in merged_quadrilaterals:
            if d == 'h':
                merged_text_height = q.aabb.w
                merged_d = 'h'
            elif d == 'v':
                merged_text_height = q.aabb.h
                merged_d = 'h'
            merged_region_imgs.append(q.get_transformed_region(image, merged_d, merged_text_height))
            
        # Save the images to temporary files for ONNX model inference
        for idx in range(len(merged_region_imgs)):
            temp_img_path = f'temp_ocr_region_{idx}.png'
            Image.fromarray(merged_region_imgs[idx]).save(temp_img_path)
            texts[idx] = self.mocr(temp_img_path)
            # Remove temporary files after processing
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

        ix = 0
        out_regions = {}
        for indices in chunks(perm, max_chunk_size):
            N = len(indices)
            widths = [region_imgs[i].shape[1] for i in indices]
            max_width = 4 * (max(widths) + 7) // 4
            region = np.zeros((N, text_height, max_width, 3), dtype = np.uint8)
            idx_keys = []
            for i, idx in enumerate(indices):
                idx_keys.append(idx)
                W = region_imgs[idx].shape[1]
                tmp = region_imgs[idx]
                region[i, :, : W, :]=tmp
                if verbose:
                    os.makedirs('result/ocrs/', exist_ok=True)
                    if quadrilaterals[idx][1] == 'v':
                        cv2.imwrite(f'result/ocrs/{ix}.png', cv2.rotate(cv2.cvtColor(region[i, :, :, :], cv2.COLOR_RGB2BGR), cv2.ROTATE_90_CLOCKWISE))
                    else:
                        cv2.imwrite(f'result/ocrs/{ix}.png', cv2.cvtColor(region[i, :, :, :], cv2.COLOR_RGB2BGR))
                ix += 1
            image_tensor = (torch.from_numpy(region).float() - 127.5) / 127.5
            image_tensor = einops.rearrange(image_tensor, 'N H W C -> N C H W')
            if self.use_gpu:
                image_tensor = image_tensor.to(self.device)
            with torch.no_grad():
                ret = self.model.infer_beam_batch(image_tensor, widths, beams_k = 5, max_seq_length = 255)
            for i, (pred_chars_index, prob, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred) in enumerate(ret):
                if prob < 0.2:
                    continue
                has_fg = (fg_ind_pred[:, 1] > fg_ind_pred[:, 0])
                has_bg = (bg_ind_pred[:, 1] > bg_ind_pred[:, 0])
                fr = AvgMeter()
                fg = AvgMeter()
                fb = AvgMeter()
                br = AvgMeter()
                bg = AvgMeter()
                bb = AvgMeter()
                for chid, c_fg, c_bg, h_fg, h_bg in zip(pred_chars_index, fg_pred, bg_pred, has_fg, has_bg) :
                    ch = self.model.dictionary[chid]
                    if ch == '<S>':
                        continue
                    if ch == '</S>':
                        break
                    if h_fg.item() :
                        fr(int(c_fg[0] * 255))
                        fg(int(c_fg[1] * 255))
                        fb(int(c_fg[2] * 255))
                    if h_bg.item() :
                        br(int(c_bg[0] * 255))
                        bg(int(c_bg[1] * 255))
                        bb(int(c_bg[2] * 255))
                    else :
                        br(int(c_fg[0] * 255))
                        bg(int(c_fg[1] * 255))
                        bb(int(c_fg[2] * 255))
                fr = min(max(int(fr()), 0), 255)
                fg = min(max(int(fg()), 0), 255)
                fb = min(max(int(fb()), 0), 255)
                br = min(max(int(br()), 0), 255)
                bg = min(max(int(bg()), 0), 255)
                bb = min(max(int(bb()), 0), 255)
                cur_region = quadrilaterals[indices[i]][0]
                if isinstance(cur_region, Quadrilateral):
                    cur_region.prob = prob
                    cur_region.fg_r = fr
                    cur_region.fg_g = fg
                    cur_region.fg_b = fb
                    cur_region.bg_r = br
                    cur_region.bg_g = bg
                    cur_region.bg_b = bb
                else:
                    cur_region.update_font_colors(np.array([fr, fg, fb]), np.array([br, bg, bb]))

                out_regions[idx_keys[i]] = cur_region

        output_regions = []
        for i, nodes in enumerate(merged_idx):
            total_logprobs = 0
            total_area = 0
            fg_r = []
            fg_g = []
            fg_b = []
            bg_r = []
            bg_g = []
            bg_b = []

            for idx in nodes:
                if idx not in out_regions:
                    continue

                region_out = out_regions[idx]
                total_logprobs += np.log(region_out.prob) * region_out.area
                total_area += region_out.area
                fg_r.append(region_out.fg_r)
                fg_g.append(region_out.fg_g)
                fg_b.append(region_out.fg_b)
                bg_r.append(region_out.bg_r)
                bg_g.append(region_out.bg_g)
                bg_b.append(region_out.bg_b)

            if total_area > 0:
                total_logprobs /= total_area
                prob = np.exp(total_logprobs)
            else:
                prob = 0.0
            fr = round(np.mean(fg_r)) if fg_r else 0
            fg = round(np.mean(fg_g)) if fg_g else 0
            fb = round(np.mean(fg_b)) if fg_b else 0
            br = round(np.mean(bg_r)) if bg_r else 0
            bg = round(np.mean(bg_g)) if bg_g else 0
            bb = round(np.mean(bg_b)) if bg_b else 0

            txt = texts[i]
            self.logger.info(f'prob: {prob} {txt} fg: ({fr}, {fg}, {fb}) bg: ({br}, {bg}, {bb})')
            cur_region = merged_quadrilaterals[i][0]
            if isinstance(cur_region, Quadrilateral):
                cur_region.text = txt
                cur_region.prob = prob
                cur_region.fg_r = fr
                cur_region.fg_g = fg
                cur_region.fg_b = fb
                cur_region.bg_r = br
                cur_region.bg_g = bg
                cur_region.bg_b = bb
            else: # TextBlock
                cur_region.text.append(txt)
                cur_region.update_font_colors(np.array([fr, fg, fb]), np.array([br, bg, bb]))
            output_regions.append(cur_region)

        if is_quadrilaterals:
            return output_regions
        return textlines