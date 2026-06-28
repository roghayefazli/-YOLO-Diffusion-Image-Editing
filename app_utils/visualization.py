import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_bboxes(image, objects):
    """
    image: np.ndarray or PIL.Image
    objects: list of dicts with keys: bbox, class, confidence
    """

    # اگر numpy بود، به PIL تبدیل کن
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)

    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = f'{obj["class"]} {obj["confidence"]:.2f}'

        # رسم bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # متن
        draw.text((x1, max(y1 - 12, 0)), label, fill="red")

    return image