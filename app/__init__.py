import warnings

from PIL import Image

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

# LangGraph currently triggers this import-time warning via langchain_core's
# default Reviver configuration before we have any serializer instance to tune.
# Keep the startup log clean until the upstream dependency switches to an
# explicit allowlist by default.
warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects` will change in a future version\..*",
    category=LangChainPendingDeprecationWarning,
)

# Allow processing very large images in preprocessing endpoints.
Image.MAX_IMAGE_PIXELS = None
