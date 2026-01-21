from .fal_FluxFlexEdit import Flux2FlexEditFal
from .fal_FluxProEdit import Flux2ProEdit_FAL
from .fal_kling_o1_image import FalKlingO1Node
from .fal_seedvr_upscale_image import FalSeedVRUpscaleImage

NODE_CLASS_MAPPINGS = {
    "Flux2FlexEditFal": Flux2FlexEditFal,
    "Flux2ProEdit_FAL": Flux2ProEdit_FAL,
    "FalKlingO1Node": FalKlingO1Node,
    "FalSeedVRUpscaleImage": FalSeedVRUpscaleImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2FlexEditFal": "FAL • FLUX.2 [flex] • Edit (up to 5 images)",
    "Flux2ProEdit_FAL": "FLUX.2 [pro] Edit (fal.ai)",
    "FalKlingO1Node": "fal.ai Kling Image O1",
    "FalSeedVRUpscaleImage": "fal.ai SeedVR2 Upscale Image",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
