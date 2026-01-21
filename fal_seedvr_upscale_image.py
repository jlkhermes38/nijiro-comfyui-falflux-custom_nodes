import os
import io
import base64
import requests
from PIL import Image
import numpy as np
import torch
import fal_client


def comfy_image_to_data_url(image_tensor):
    """
    ComfyUI IMAGE est typiquement (B, H, W, C) float32 [0..1].
    Convertit la 1ère image du batch en data URI PNG base64.
    """
    arr = image_tensor[0].detach().cpu().numpy()  # (H, W, C)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def pil_to_comfy(pil_img):
    """
    PIL -> ComfyUI tensor (B, H, W, C) float32 [0..1]
    """
    pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img).astype(np.float32) / 255.0  # (H, W, C)
    arr = np.expand_dims(arr, axis=0)  # (B, H, W, C)
    return torch.from_numpy(arr)


class FalSeedVRUpscaleImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),

                "upscale_mode": (["factor", "target"], {"default": "factor"}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1}),
                "target_resolution": (["720p", "1080p", "1440p", "2160p"], {"default": "1080p"}),

                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "step": 1}),
                "noise_scale": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "output_format": (["jpg", "png", "webp"], {"default": "jpg"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "fal.ai / SeedVR"

    def run(
        self,
        api_key,
        image,
        upscale_mode,
        upscale_factor,
        target_resolution,
        seed,
        noise_scale,
        output_format,
    ):
        os.environ["FAL_KEY"] = api_key

        image_url = comfy_image_to_data_url(image)

        args = {
            "image_url": image_url,
            "upscale_mode": upscale_mode,
            "noise_scale": float(noise_scale),
            "output_format": output_format,
            "seed": int(seed),
        }

        if upscale_mode == "factor":
            args["upscale_factor"] = float(upscale_factor)
        else:
            args["target_resolution"] = target_resolution

        result = fal_client.run("fal-ai/seedvr/upscale/image", arguments=args)

        image_obj = result.get("image")
        if not image_obj or "url" not in image_obj:
            raise RuntimeError(f"Réponse inattendue de SeedVR Upscale Image: {result}")

        url = image_obj["url"]
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        pil_img = Image.open(io.BytesIO(r.content))

        return (pil_to_comfy(pil_img),)


NODE_CLASS_MAPPINGS = {
    "FalSeedVRUpscaleImage": FalSeedVRUpscaleImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalSeedVRUpscaleImage": "fal.ai SeedVR2 Upscale Image",
}
