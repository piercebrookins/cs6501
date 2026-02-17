from __future__ import annotations

from pathlib import Path

from PIL import Image


def resize_image_in_place(image_path: str, max_width: int = 1024, quality: int = 85) -> str:
    """Resize an image if needed and save a compressed copy beside it."""
    src = Path(image_path)
    with Image.open(src) as img:
        if img.width <= max_width:
            return str(src)

        ratio = max_width / float(img.width)
        new_size = (max_width, int(img.height * ratio))
        resized = img.resize(new_size)

        out_path = src.with_name(f"{src.stem}_resized{src.suffix}")
        resized.save(out_path, quality=quality, optimize=True)
        return str(out_path)
