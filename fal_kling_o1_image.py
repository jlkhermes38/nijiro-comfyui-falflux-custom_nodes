import os
import io
import base64
import requests
from PIL import Image
import numpy as np
import torch
import fal_client   # ← Maintenant qu’il est installé


def comfy_image_to_data_url(image_tensor, min_size=300):
    """
    image_tensor : torch.Tensor au format (B, H, W, C) avec valeurs 0–1 (format standard ComfyUI)
    On le convertit en PNG base64, en s’assurant que la taille min est >= min_size.
    """
    # B, H, W, C -> H, W, C
    arr = image_tensor[0].detach().cpu().numpy()  # H, W, C
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="RGB")

    # Vérifie la taille minimale pour Kling (300x300)
    w, h = img.size
    if w < min_size or h < min_size:
        scale = max(min_size / w, min_size / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def img_to_tensor(pil_img):
    """
    Convertit une image PIL en tensor ComfyUI (B, H, W, C) float32 [0,1].
    """
    pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img).astype(np.float32) / 255.0  # H, W, C
    arr = np.expand_dims(arr, axis=0)  # B, H, W, C
    t = torch.from_numpy(arr)
    return t


class FalKlingO1Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"default": "Put @Image1 into @Image2"})
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "fal.ai"

    def run(self, api_key, prompt,
            image_1=None, image_2=None, image_3=None, image_4=None, image_5=None):

        os.environ["FAL_KEY"] = api_key

        # Images → data URL base64
        imgs = [image_1, image_2, image_3, image_4, image_5]
        image_urls = [comfy_image_to_data_url(i) for i in imgs if i is not None]

        if not image_urls:
            raise ValueError("Kling O1 requiert au moins une image d'entrée.")

        # Appel API via fal_client
        data = fal_client.run(
            "fal-ai/kling-image/o1",
            arguments={
                "prompt": prompt,
                "image_urls": image_urls,
                "resolution": "1K",
                "num_images": 1,
                "aspect_ratio": "auto",
                "output_format": "png",
            }
        )

        url = data["images"][0]["url"]

        # Téléchargement de l’image
        r = requests.get(url)
        r.raise_for_status()
        pil = Image.open(io.BytesIO(r.content))

        return (img_to_tensor(pil),)


NODE_CLASS_MAPPINGS = {"FalKlingO1Node": FalKlingO1Node}
NODE_DISPLAY_NAME_MAPPINGS = {"FalKlingO1Node": "fal.ai Kling Image O1"}
