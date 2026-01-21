import os
import io
import requests
import numpy as np
import torch
from PIL import Image

import fal_client


MODEL_ID = "fal-ai/flux-2-flex/edit"

IMAGE_SIZE_ENUM = [
    "auto",
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
    "custom",
]

OUTPUT_FORMAT_ENUM = ["jpeg", "png"]


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("Expected a torch.Tensor for IMAGE input.")

    if image_tensor.ndim != 4 or image_tensor.shape[-1] not in (3, 4):
        raise ValueError("Expected IMAGE tensor with shape [B,H,W,3] or [B,H,W,4].")

    img = image_tensor[0].detach().cpu().float().clamp(0, 1).numpy()
    img = (img * 255.0).round().astype(np.uint8)

    if img.shape[-1] == 4:
        pil = Image.fromarray(img, mode="RGBA").convert("RGB")
    else:
        pil = Image.fromarray(img, mode="RGB")
    return pil


def _pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    arr = np.asarray(pil).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # [1,H,W,3]
    return torch.from_numpy(arr)


def _download_image(url: str, timeout: int = 120) -> Image.Image:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


class Flux2FlexEditFal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),  # note: "password" pas toujours supporté
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image1": ("IMAGE",),
                "image_size": (IMAGE_SIZE_ENUM, {"default": "auto"}),
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "custom_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "output_format": (OUTPUT_FORMAT_ENUM, {"default": "jpeg"}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("image", "info", "used_seed")
    FUNCTION = "run"
    CATEGORY = "image/fal"

    def run(
        self,
        api_key,
        prompt,
        image1,
        image_size,
        custom_width,
        custom_height,
        seed,
        guidance_scale,
        num_inference_steps,
        output_format,
        enable_safety_checker,
        enable_prompt_expansion,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
    ):
        # --- API KEY ---
        if api_key and str(api_key).strip():
            os.environ["FAL_KEY"] = str(api_key).strip()
        elif os.environ.get("FAL_KEY"):
            pass
        else:
            raise RuntimeError("No FAL API key provided (node api_key or env var FAL_KEY).")

        # --- Collect up to 5 images ---
        pil_images = []
        for img in [image1, image2, image3, image4, image5]:
            if img is None:
                continue
            pil_images.append(_tensor_to_pil(img))

        # --- Upload to fal CDN ---
        image_urls = []
        for pil in pil_images:
            url = fal_client.upload_image(pil, format="png")
            image_urls.append(url)

        # --- Image size payload ---
        if image_size == "custom":
            image_size_payload = {"width": int(custom_width), "height": int(custom_height)}
        else:
            image_size_payload = image_size

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "image_size": image_size_payload,
            "enable_prompt_expansion": bool(enable_prompt_expansion),
            "enable_safety_checker": bool(enable_safety_checker),
            "output_format": output_format,
            "guidance_scale": float(guidance_scale),
            "num_inference_steps": int(num_inference_steps),
        }
        if seed is not None and int(seed) >= 0:
            arguments["seed"] = int(seed)

        # --- Call fal ---
        result = fal_client.subscribe(
            MODEL_ID,
            arguments=arguments,
            with_logs=False,
        )

        out_images = result.get("images", [])
        if not out_images:
            raise RuntimeError(f"No images returned. Raw result: {result}")

        out_url = out_images[0].get("url")
        if not out_url:
            raise RuntimeError(f"Missing output url. Raw result: {result}")

        out_pil = _download_image(out_url)
        out_tensor = _pil_to_tensor(out_pil)

        used_seed = int(result.get("seed", -1))

        info = (
            f"model={MODEL_ID}\n"
            f"input_images={len(image_urls)}\n"
            f"output_url={out_url}\n"
            f"seed={used_seed}\n"
            f"guidance_scale={guidance_scale}\n"
            f"num_inference_steps={num_inference_steps}\n"
            f"output_format={output_format}\n"
            f"enable_safety_checker={enable_safety_checker}\n"
            f"enable_prompt_expansion={enable_prompt_expansion}\n"
        )

        return (out_tensor, info, used_seed)


NODE_CLASS_MAPPINGS = {
    "Flux2FlexEditFal": Flux2FlexEditFal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2FlexEditFal": "FAL • FLUX.2 [flex] • Edit (up to 5 images)",
}
