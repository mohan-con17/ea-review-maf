from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Tuple, List

from PIL import Image, UnidentifiedImageError

from app.domain.review_models import ReviewSessionContext, PreprocessedImage

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    ImagePreprocessor
    -----------------
    Non-LLM image preprocessing.

    Responsibilities:
    - Decode base64 / data URL from ctx.metadata["arch_img_url"]
    - Inspect image dimensions
    - Split large images into tiles for better local detail
    - Store PreprocessedImage in ctx.preprocessed_image

    Safety guarantees:
    - MUST NEVER throw
    - MUST tolerate missing metadata, invalid base64,
      corrupted images, and PIL errors
    """
    async def run(self, ctx: ReviewSessionContext):
        review_id = ctx.review_id

        try:
            metadata = ctx.metadata or {}
            arch_img_url = metadata.get("arch_img_url")

            if not arch_img_url:
                raise ValueError("arch_img_url is missing")

            image_bytes, content_type, ext = self._parse_data_url_to_bytes(arch_img_url)
            tiles, width, height, tiles_x, tiles_y = self._split_image_into_tiles(image_bytes)

            ctx.preprocessed_image = PreprocessedImage(
                content_type=content_type,
                ext=ext,
                width=width,
                height=height,
                tiles=tiles,
                tiles_x=tiles_x,
                tiles_y=tiles_y,
            )

            return ctx.preprocessed_image

        except Exception as e:
            # 🔴 HARD FAIL SIGNAL
            logger.exception("[%s] ImagePreprocessor failed", review_id)
            ctx.preprocessed_image = None

            class _Fail:
                error = f"Image preprocessing failed: {str(e)}"

            return _Fail()


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_data_url_to_bytes(self, data_url: str) -> Tuple[bytes, str, str]:
        """
        Convert a data URL or raw base64 string into image bytes.

        Accepts:
        - data:image/png;base64,...
        - raw base64 string (assumes PNG)

        Raises ValueError on any decoding issue.
        """
        if not isinstance(data_url, str):
            raise ValueError("arch_img_url must be a string")

        if data_url.startswith("data:"):
            header, b64_part = data_url.split(",", 1)

            if ";base64" not in header:
                raise ValueError("Data URL is not base64-encoded")

            content_type = header[len("data:"): header.index(";base64")]
            base64_str = b64_part.strip()
        else:
            # Fallback: raw base64 with assumed PNG
            content_type = "image/png"
            base64_str = data_url.strip()

        image_bytes = base64.b64decode(base64_str, validate=True)
        if not image_bytes:
            raise ValueError("Decoded image is empty")

        content_type_lower = content_type.lower()
        if content_type_lower in ("image/jpeg", "image/jpg"):
            ext = "jpg"
        elif content_type_lower == "image/webp":
            ext = "webp"
        else:
            ext = "png"

        return image_bytes, content_type, ext

    def _split_image_into_tiles(
        self,
        image_bytes: bytes,
    ) -> Tuple[List[bytes], int, int, int, int]:
        """
        Split image into tiles for improved local detail.

        Strategy:
        - Small images (<1400px on both axes): no tiling
        - Medium images: 2x2 tiles
        - Very large images (>=2600px max dimension): 3x3 tiles

        Tiles are saved as PNG to preserve fine text and lines.
        """
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                width, height = img.size

                # No tiling for small images
                if width < 1400 and height < 1400:
                    logger.info(
                        "Image size (%dx%d) considered small; no tiling applied",
                        width,
                        height,
                    )
                    return [image_bytes], width, height, 1, 1

                max_dim = max(width, height)
                tiles_x = tiles_y = 3 if max_dim >= 2600 else 2

                tile_w = width // tiles_x
                tile_h = height // tiles_y

                tiles: List[bytes] = []
                for i in range(tiles_x):
                    for j in range(tiles_y):
                        left = i * tile_w
                        upper = j * tile_h
                        right = (
                            (i + 1) * tile_w if i < tiles_x - 1 else width
                        )
                        lower = (
                            (j + 1) * tile_h if j < tiles_y - 1 else height
                        )

                        crop = img.crop((left, upper, right, lower))
                        buf = BytesIO()
                        crop.save(buf, format="PNG")
                        tiles.append(buf.getvalue())

                logger.info(
                    "Image split into %d tiles (%dx%d). Original size=%dx%d",
                    len(tiles),
                    tiles_x,
                    tiles_y,
                    width,
                    height,
                )

                return tiles, width, height, tiles_x, tiles_y

        except (UnidentifiedImageError, OSError) as e:
            raise ValueError("Unsupported or corrupted image") from e
