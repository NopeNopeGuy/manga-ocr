from typing import List
from PIL import Image

from .common import CommonUpscaler, OfflineUpscaler
from .waifu2x import Waifu2xUpscaler
from .esrgan import ESRGANUpscaler
from .esrgan_pytorch import ESRGANUpscalerPytorch
from .anime4k_onnx import Anime4K_ONNXUpscaler
from ..config import Upscaler

UPSCALERS = {
    Upscaler.waifu2x: Waifu2xUpscaler,
    Upscaler.esrgan: ESRGANUpscaler,
    Upscaler.upscler4xultrasharp: ESRGANUpscalerPytorch,
    Upscaler.anime4k_onnx: Anime4K_ONNXUpscaler,
}
upscaler_cache = {}

def get_upscaler(key: Upscaler, *args, **kwargs) -> CommonUpscaler:
    if key not in UPSCALERS:
        raise ValueError(f'Could not find upscaler for: "{key}". Choose from the following: %s' % ','.join(UPSCALERS))
    if not upscaler_cache.get(key):
        upscaler = UPSCALERS[key]
        upscaler_cache[key] = upscaler(*args, **kwargs)
    return upscaler_cache[key]

async def prepare(upscaler_key: Upscaler):
    upscaler = get_upscaler(upscaler_key)
    if isinstance(upscaler, OfflineUpscaler):
        await upscaler.download()

async def dispatch(upscaler_key: Upscaler, image_batch: List[Image.Image], upscale_ratio: int, device: str = 'cpu') -> List[Image.Image]:
    if upscale_ratio == 1:
        return image_batch
    if upscaler_key not in UPSCALERS:
        raise ValueError(f'Could not find upscaler for: "{upscaler_key}". Choose from the following: %s' % ','.join(UPSCALERS))
    upscaler_class = UPSCALERS[upscaler_key]
    upscaler = upscaler_class()
    if isinstance(upscaler, OfflineUpscaler):
        await upscaler.load(device)
    return await upscaler.upscale(image_batch, upscale_ratio)

async def unload(upscaler_key: Upscaler):
    upscaler_cache.pop(upscaler_key, None)
