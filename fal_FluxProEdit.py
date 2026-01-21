import os
import io
import tempfile
import requests
import numpy as np
from PIL import Image

import torch

try:
    import fal_client
except Exception as e:
    fal_client = None


def _ensure_fal_client():
    if fal_client is None:
        raise RuntimeError(
            "fal-client n'est pas installé. Installez-le avec: pip install fal-client"
        )


def _set_fal_api_key(api_key: str):
    # --- EXACTEMENT ta logique + compatibilité fal-client récent ---
    if api_key and str(api_key).strip():
        key = str(api_key).strip()
        os.environ["FAL_KEY"] = key
        fal_client.api_key = key

    elif os.environ.get("FAL_KEY"):
        fal_client.api_key = os.environ.get("FAL_KEY")

    else:
        raise RuntimeError(
            "No FAL API key provided (node api_key or env var FAL_KEY)."
        )



def _comfy_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    ComfyUI IMAGE: torch float32 [B,H,W,C] in 0..1 (souvent).
    On prend la première image du batch.
    """
    if image_tensor is None:
        return None

    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("Entrée image ComfyUI invalide (attendu torch.Tensor).")

    if image_tensor.dim() == 4:
        img = image_tensor[0]
    elif image_tensor.dim() == 3:
        img = image_tensor
    else:
        raise ValueError(f"Shape d'image inattendue: {tuple(image_tensor.shape)}")

    img = img.detach().cpu().float().clamp(0, 1).numpy()
    img = (img * 255.0).round().astype(np.uint8)

    # img: [H,W,C]
    if img.shape[-1] == 4:
        return Image.fromarray(img, mode="RGBA")
    return Image.fromarray(img, mode="RGB")


def _pil_to_comfy_image(pil_img: Image.Image) -> torch.Tensor:
    """
    Retour ComfyUI IMAGE: torch float32 [1,H,W,3] in 0..1
    """
    if pil_img.mode not in ("RGB", "RGBA"):
        pil_img = pil_img.convert("RGB")
    if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")

    arr = np.array(pil_img).astype(np.float32) / 255.0  # [H,W,3]
    t = torch.from_numpy(arr)[None, ...]  # [1,H,W,3]
    return t


def _upload_pil_to_fal(pil_img: Image.Image, output_format: str) -> str:
    """
    Upload via fal_client.upload_file() en écrivant un fichier temporaire.
    """
    _ensure_fal_client()

    suffix = ".png" if output_format == "png" else ".jpg"
    fmt = "PNG" if output_format == "png" else "JPEG"

    # JPEG n'aime pas l'alpha
    if fmt == "JPEG" and pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name

    try:
        pil_img.save(tmp_path, format=fmt, quality=95)
        url = fal_client.upload_file(tmp_path)
        return url
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _parse_image_size(image_size_mode: str, width: int, height: int):
    """
    image_size accepte soit un enum, soit un objet {width,height}. :contentReference[oaicite:1]{index=1}
    """
    if image_size_mode == "custom":
        w = int(width)
        h = int(height)
        if w <= 0 or h <= 0:
            raise ValueError("Custom width/height doivent être > 0.")
        return {"width": w, "height": h}

    # enums côté fal: auto, square_hd, square, portrait_4_3, portrait_16_9, landscape_4_3, landscape_16_9 :contentReference[oaicite:2]{index=2}
    return image_size_mode


class Flux2ProEdit_FAL:
    """
    ComfyUI Node: fal-ai/flux-2-pro/edit
    - Jusqu'à 5 images (optionnelles) -> upload fal -> image_urls[]
    - prompt
    - output_format (jpeg/png)
    - enable_safety_checker
    - image_size (enum + custom WxH)
    - api_key (avec fallback env FAL_KEY)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "image_size_mode": ([
                    "auto",
                    "square_hd",
                    "square",
                    "portrait_4_3",
                    "portrait_16_9",
                    "landscape_4_3",
                    "landscape_16_9",
                    "custom",
                ], {"default": "auto"}),
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "custom_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**31 - 1
                }),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "result_url")
    FUNCTION = "run"
    CATEGORY = "fal.ai / FLUX"

    def run(
        self,
        prompt: str,
        output_format: str,
        enable_safety_checker: bool,
        image_size_mode: str,
        custom_width: int,
        custom_height: int,
        seed: int,
        api_key: str,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
    ):
        _ensure_fal_client()
        _set_fal_api_key(api_key)

        # Collecte images (jusqu'à 5)
        images_in = [image_1, image_2, image_3, image_4, image_5]
        pil_images = []
        for im in images_in:
            if im is None:
                continue
            pil_images.append(_comfy_image_to_pil(im))

        if len(pil_images) == 0:
            raise RuntimeError("Aucune image fournie. Le endpoint fal-ai/flux-2-pro/edit requiert au moins 1 image.")  # :contentReference[oaicite:3]{index=3}

        # Upload vers fal.media -> image_urls
        image_urls = []
        for pimg in pil_images:
            image_urls.append(_upload_pil_to_fal(pimg, output_format))

        # image_size (enum ou custom object)
        image_size = _parse_image_size(image_size_mode, custom_width, custom_height)

        # Appel modèle
        # Paramètres conformes au schéma: prompt, image_urls, image_size, enable_safety_checker, output_format :contentReference[oaicite:4]{index=4}
        arguments={
                "prompt": prompt,
                "image_urls": image_urls,
                "image_size": image_size,
                "enable_safety_checker": bool(enable_safety_checker),
                "output_format": output_format,
        }
        # Seed optionnel
        if seed is not None and int(seed) >= 0:
            arguments["seed"] = int(seed)
        
        result = fal_client.subscribe(
            "fal-ai/flux-2-pro/edit",
            arguments=arguments,
        )

        # result attendu: {"images":[{"url":...}], "seed": ...} :contentReference[oaicite:5]{index=5}
        data = result if isinstance(result, dict) else getattr(result, "data", None) or {}
        images = data.get("images") or []
        if not images:
            raise RuntimeError(f"Réponse fal inattendue (pas d'images): {data}")

        out_url = images[0].get("url")
        if not out_url:
            raise RuntimeError(f"Réponse fal inattendue (url manquante): {images[0]}")

        # Download image -> Comfy tensor
        r = requests.get(out_url, timeout=120)
        r.raise_for_status()
        out_pil = Image.open(io.BytesIO(r.content)).convert("RGB")
        out_tensor = _pil_to_comfy_image(out_pil)

        return (out_tensor, out_url)


NODE_CLASS_MAPPINGS = {
    "Flux2ProEdit_FAL": Flux2ProEdit_FAL
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2ProEdit_FAL": "FLUX.2 [pro] Edit (fal.ai)"
}
