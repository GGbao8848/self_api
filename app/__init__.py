from PIL import Image

# Allow processing very large images in preprocessing endpoints.
Image.MAX_IMAGE_PIXELS = None
