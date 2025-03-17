import os
from typing import List
from PIL import Image
import numpy as np
import onnxruntime

from manga_translator.upscaling.common import CommonUpscaler, OfflineUpscaler


class Anime4K_ONNXUpscaler(OfflineUpscaler, CommonUpscaler):
    _VALID_UPSCALE_RATIOS = [2] # Example ratios, adjust as needed
    name = "anime4k_onnx"

    def __init__(self, model_path="models/upscaling/Anime4K_Upscale_GAN_x2_M.onnx"):
        super().__init__()
        self.model_path = model_path
        self.ort_session = None
        self._loaded = False

    async def download(self):
        # Model is assumed to be already present, no download needed
        pass

    async def _load(self, device: str = 'cpu'):
        providers = ['OpenVINOExecutionProvider','CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']
        self.ort_session = onnxruntime.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        print(f"Anime4K ONNX upscaler loaded on {device}")
        self._loaded = True

    async def _unload(self):
        self.ort_session = None
        self._loaded = False

    async def _infer(self, image_batch: List[Image.Image], upscale_ratio: float) -> List[Image.Image]:
        if self.ort_session is None:
            raise Exception("Model not loaded. Call load() first.")

        upscaled_images = []
        for image in image_batch:
            img_np = np.array(image).astype(np.float16) / 255.0
            if img_np.ndim == 2:  # Handle grayscale images
                img_np = np.stack([img_np, img_np, img_np], axis=-1)  # Convert grayscale to RGB
            img_np = np.transpose(img_np, (2, 0, 1))  # HWC to CHW
            img_np = np.expand_dims(img_np, axis=0)     # Add batch dimension

            ort_inputs = {self.input_name: img_np}
            ort_outputs = self.ort_session.run([self.output_name], ort_inputs)
            output_np = ort_outputs[0]

            output_np = np.squeeze(output_np, axis=0)  # Remove batch dimension
            output_np = np.transpose(output_np, (1, 2, 0))  # CHW to HWC
            output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
            upscaled_image = Image.fromarray(output_np)
            upscaled_images.append(upscaled_image)
        return upscaled_images
