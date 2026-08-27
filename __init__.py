from .multishot_stills import NODE_CLASS_MAPPINGS as _STILL_MAP
from .multishot_stills import NODE_DISPLAY_NAME_MAPPINGS as _STILL_NAMES
from .ltx_multiscene import NODE_CLASS_MAPPINGS as _LTX_MAP
from .ltx_multiscene import NODE_DISPLAY_NAME_MAPPINGS as _LTX_NAMES
from .ltx_multiscene_msr import NODE_CLASS_MAPPINGS as _MSR_MAP
from .ltx_multiscene_msr import NODE_DISPLAY_NAME_MAPPINGS as _MSR_NAMES

NODE_CLASS_MAPPINGS = {**_STILL_MAP, **_LTX_MAP, **_MSR_MAP}
NODE_DISPLAY_NAME_MAPPINGS = {**_STILL_NAMES, **_LTX_NAMES, **_MSR_NAMES}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
