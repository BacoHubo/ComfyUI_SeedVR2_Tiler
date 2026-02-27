"""
SeedVR2 Tile Splitter & Stitcher
=================================
Add-on for ComfyUI-SeedVR2_VideoUpscaler that enables tiled upscaling of
images that would otherwise exceed available VRAM.

Tiling algorithm inspired by the Divide and Conquer Node Suite (Steudio):
  - Compute the minimum NxM grid such that each tile ≤ tile_size_mp
  - Among all grids with the same (minimum) tile count, pick the most square
    one relative to the image aspect ratio
  - Resize the image to an exact canvas (cols×tile_w × rows×tile_h) so every
    tile is IDENTICAL in size — no partial edge tiles
  - Adjacent tiles overlap by overlap_percent on each edge

Workflow:
  [Image]
    → [SeedVR2 Tile Splitter]  →  IMAGE batch  →  [SeedVR2 node]
                               →  TILE_METADATA →  [SeedVR2 Tile Stitcher]
    [SeedVR2 node output]      →  [SeedVR2 Tile Stitcher]  →  final IMAGE
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _align(value: float, multiple: int = 8) -> int:
    """Round up to nearest multiple of `multiple` (required by VAE / SeedVR2)."""
    return max(multiple, int(math.ceil(value / multiple)) * multiple)


def _resize(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Bilinear resize. img: (H, W, C) -> (h, w, C)."""
    x = img.permute(2, 0, 1).unsqueeze(0).float()
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.squeeze(0).permute(1, 2, 0).to(img.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Core grid algorithm
# ─────────────────────────────────────────────────────────────────────────────

def compute_tile_grid(
    img_w: int,
    img_h: int,
    tile_mp: float,
    overlap_fraction: float,
) -> Dict[str, Any]:
    """
    Find the optimal cols x rows tile grid for the given image.

    Rules:
      1. Every tile must be <= tile_mp megapixels (after 8-px alignment).
      2. The canvas (cols*tile_w x rows*tile_h) must cover the full image.
      3. Adjacent tiles overlap by overlap_fraction of the tile dimension.
      4. Among all grids with the minimum viable tile count, prefer the most
         square grid (cols/rows closest to the image aspect ratio).
      5. A slightly larger tile count is accepted if it gives a much squarer
         grid (combined score = n/min_n + squareness).

    Returns a dict with:
      cols, rows, tile_w, tile_h, overlap_w, overlap_h,
      stride_w, stride_h, canvas_w, canvas_h, n_tiles
    """
    target_px = tile_mp * 1_000_000
    aspect    = img_w / img_h

    def _make_grid(cols: int, rows: int) -> Dict[str, Any]:
        # Tile size formula (see derivation in docstring above):
        #   canvas_w = tile_w * (cols - overlap_fraction*(cols-1))
        #   => tile_w = img_w / denom_w
        denom_w = cols - overlap_fraction * (cols - 1)
        denom_h = rows - overlap_fraction * (rows - 1)

        tile_w = _align(img_w / denom_w, 8)
        tile_h = _align(img_h / denom_h, 8)

        overlap_w = min(_align(tile_w * overlap_fraction, 8), tile_w - 8)
        overlap_h = min(_align(tile_h * overlap_fraction, 8), tile_h - 8)

        stride_w = tile_w - overlap_w
        stride_h = tile_h - overlap_h

        canvas_w = stride_w * (cols - 1) + tile_w
        canvas_h = stride_h * (rows - 1) + tile_h

        # Guarantee canvas covers image — alignment can undershoot by 1 step
        if canvas_w < img_w:
            tile_w   += 8
            overlap_w = min(_align(tile_w * overlap_fraction, 8), tile_w - 8)
            stride_w  = tile_w - overlap_w
            canvas_w  = stride_w * (cols - 1) + tile_w

        if canvas_h < img_h:
            tile_h   += 8
            overlap_h = min(_align(tile_h * overlap_fraction, 8), tile_h - 8)
            stride_h  = tile_h - overlap_h
            canvas_h  = stride_h * (rows - 1) + tile_h

        return dict(
            cols=cols, rows=rows,
            tile_w=tile_w, tile_h=tile_h,
            overlap_w=overlap_w, overlap_h=overlap_h,
            stride_w=stride_w, stride_h=stride_h,
            canvas_w=canvas_w, canvas_h=canvas_h,
            n_tiles=cols * rows,
        )

    def _squareness(cols: int, rows: int) -> float:
        """
        Score how well tile aspect ratio matches image aspect ratio.
        0 = tile shape perfectly matches image; higher = more distorted.
        Scores TILE aspect (tile_w/tile_h vs img aspect), NOT grid ratio
        (cols/rows) -- the latter caused portrait images to get landscape tiles.
        """
        g = _make_grid(cols, rows)
        tile_aspect = g["tile_w"] / g["tile_h"]
        return abs(math.log(tile_aspect / aspect))

    # Collect all valid (cols, rows) pairs up to 2x the minimum viable count
    candidates: List[Tuple[int, float, int, int, Dict]] = []

    for n in range(1, 512):
        for c in range(1, n + 1):
            if n % c != 0:
                continue
            r = n // c
            g = _make_grid(c, r)
            if g["tile_w"] * g["tile_h"] <= target_px * 1.15:
                candidates.append((n, _squareness(c, r), c, r, g))

        if candidates:
            min_valid = candidates[0][0]
            if n >= min_valid * 2 + 4:
                break

    if not candidates:
        g = _make_grid(1, 1)
        g["squareness"] = 0.0
        return g

    # Score: penalise extra tiles + poor squareness equally
    min_n      = candidates[0][0]
    best       = None
    best_score = float("inf")

    for n, sq, c, r, g in candidates:
        score = (n / min_n) + sq
        if score < best_score:
            best_score     = score
            g["squareness"] = sq
            best           = g

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 – Tile Splitter
# ─────────────────────────────────────────────────────────────────────────────

class SeedVR2TileSplitter:
    """
    Splits a single image into a uniform NxM grid of overlapping tiles,
    each within `tile_size_mp` megapixels.

    The image is resized to an exact canvas before splitting so every tile
    sent to SeedVR2 is the same size.  No partial or oversized edge tiles.

    Outputs
    -------
    tiles         : IMAGE batch (N, tile_h, tile_w, C) — connect to SeedVR2
    tile_metadata : TILE_METADATA — pass directly to Tile Stitcher
    """

    CATEGORY    = "image/upscaling"
    FUNCTION    = "split"
    RETURN_TYPES  = ("IMAGE", "TILE_METADATA", "INT")
    RETURN_NAMES  = ("tiles", "tile_metadata", "resolution")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_size_mp": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1,
                        "display": "number",
                        "tooltip": (
                            "Maximum size of each tile in megapixels. "
                            "The grid is chosen so every tile is AT MOST this size. "
                            "Lower = less VRAM per SeedVR2 pass."
                        ),
                    },
                ),
                "tile_upscale_mp": (
                    "FLOAT",
                    {
                        "default": 2.0, "min": 0.1, "max": 64.0, "step": 0.1,
                        "display": "number",
                        "tooltip": (
                            "Desired output size of each upscaled tile in megapixels. "
                            "Used to compute the resolution hint for SeedVR2. "
                            "Typically 2x your tile_size_mp."
                        ),
                    },
                ),
                "overlap_percent": (
                    "FLOAT",
                    {
                        "default": 10.0, "min": 0.0, "max": 40.0, "step": 1.0,
                        "display": "number",
                        "tooltip": (
                            "Overlap between adjacent tiles as a % of tile size. "
                            "10% is a safe default. Increase to 15-20% if seams appear."
                        ),
                    },
                ),
                "feather_blend": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                        "display": "number",
                        "tooltip": (
                            "Fraction of the overlap zone used for blending during stitch. "
                            "1.0 = smooth linear feather. 0.0 = hard cut."
                        ),
                    },
                ),
            }
        }

    def split(self, image, tile_size_mp, tile_upscale_mp, overlap_percent, feather_blend):
        # Normalise to (H, W, C)
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(
                    f"Tile Splitter expects a single image (batch=1), got {image.shape[0]}."
                )
            img = image[0]
        else:
            img = image

        orig_H, orig_W, C = img.shape
        overlap_fraction  = overlap_percent / 100.0

        grid     = compute_tile_grid(orig_W, orig_H, tile_size_mp, overlap_fraction)
        cols     = grid["cols"];     rows     = grid["rows"]
        tile_w   = grid["tile_w"];   tile_h   = grid["tile_h"]
        overlap_w= grid["overlap_w"]; overlap_h= grid["overlap_h"]
        stride_w = grid["stride_w"]; stride_h = grid["stride_h"]
        canvas_w = grid["canvas_w"]; canvas_h = grid["canvas_h"]

        # Resize image to exact canvas
        if canvas_w != orig_W or canvas_h != orig_H:
            canvas_img = _resize(img, canvas_h, canvas_w)
        else:
            canvas_img = img

        # Extract tiles (row-major)
        tiles: List[torch.Tensor] = []
        positions: List[Tuple[int, int]] = []

        for row in range(rows):
            for col in range(cols):
                x0 = col * stride_w
                y0 = row * stride_h
                tiles.append(canvas_img[y0 : y0 + tile_h, x0 : x0 + tile_w, :])
                positions.append((x0, y0))

        tiles_tensor = torch.stack(tiles, dim=0)  # (N, tile_h, tile_w, C)

        # resolution hint for SeedVR2
        tile_aspect    = tile_w / tile_h
        out_h_raw      = math.sqrt(tile_upscale_mp * 1_000_000 / tile_aspect)
        resolution = _align(max(out_h_raw * tile_aspect, out_h_raw), 8)

        meta: Dict[str, Any] = {
            "orig_w": orig_W, "orig_h": orig_H, "orig_c": C,
            "canvas_w": canvas_w, "canvas_h": canvas_h,
            "cols": cols, "rows": rows, "n_tiles": cols * rows,
            "tile_w": tile_w, "tile_h": tile_h,
            "overlap_w": overlap_w, "overlap_h": overlap_h,
            "stride_w": stride_w, "stride_h": stride_h,
            "positions": positions,
            "feather_blend": feather_blend,
            "resolution": resolution,
        }

        print(
            f"\n[SeedVR2 Tile Splitter]"
            f"\n  Original : {orig_W}x{orig_H}"
            f"\n  Canvas   : {canvas_w}x{canvas_h}"
            f"{'  (resized)' if (canvas_w != orig_W or canvas_h != orig_H) else '  (exact)'}"
            f"\n  Grid     : {cols}x{rows} = {cols*rows} tiles"
            f"\n  Tile     : {tile_w}x{tile_h} = {tile_w*tile_h/1e6:.2f} MP"
            f"\n  Overlap  : {overlap_w}x{overlap_h} px  |  Stride: {stride_w}x{stride_h} px"
            f"\n  SeedVR2 resolution hint: {resolution}"
        )

        return (tiles_tensor, meta, resolution)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 – Tile Stitcher
# ─────────────────────────────────────────────────────────────────────────────

class SeedVR2TileStitcher:
    """
    Receives the upscaled tile batch (SeedVR2 output) + TILE_METADATA,
    blends overlapping regions, and returns the final stitched image.

    Actual upscale factor is detected at runtime, so the node works even if
    SeedVR2 produced a different resolution than suggested.
    """

    CATEGORY    = "image/upscaling"
    FUNCTION    = "stitch"
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscaled_tiles": ("IMAGE",),
                "tile_metadata":  ("TILE_METADATA",),
            }
        }

    def stitch(self, upscaled_tiles: torch.Tensor, tile_metadata: dict):
        meta = tile_metadata

        orig_w   = meta["orig_w"];    orig_h   = meta["orig_h"]
        canvas_w = meta["canvas_w"];  canvas_h = meta["canvas_h"]
        cols     = meta["cols"];      rows     = meta["rows"]
        n_tiles  = meta["n_tiles"]
        tile_w   = meta["tile_w"];    tile_h   = meta["tile_h"]
        overlap_w= meta["overlap_w"]; overlap_h= meta["overlap_h"]
        stride_w = meta["stride_w"];  stride_h = meta["stride_h"]
        positions = meta["positions"]
        feather_blend: float = meta["feather_blend"]

        if upscaled_tiles.shape[0] != n_tiles:
            raise ValueError(
                f"Expected {n_tiles} upscaled tiles but received "
                f"{upscaled_tiles.shape[0]}. Check batch order is unchanged."
            )

        up_h = upscaled_tiles.shape[1]
        up_w = upscaled_tiles.shape[2]
        C    = upscaled_tiles.shape[3]

        scale_x = up_w / tile_w
        scale_y = up_h / tile_h

        out_canvas_w = round(canvas_w * scale_x)
        out_canvas_h = round(canvas_h * scale_y)

        feather_px_x = max(1, round(overlap_w * scale_x * feather_blend))
        feather_px_y = max(1, round(overlap_h * scale_y * feather_blend))

        print(
            f"\n[SeedVR2 Tile Stitcher]"
            f"\n  Scale  : {scale_x:.3f}x  x  {scale_y:.3f}x"
            f"\n  Canvas : {out_canvas_w}x{out_canvas_h}"
            f"\n  Tiles  : {n_tiles} ({cols}x{rows})"
            f"\n  Feather: {feather_px_x}x{feather_px_y} px"
        )

        device = upscaled_tiles.device
        dtype  = upscaled_tiles.dtype

        canvas  = torch.zeros(out_canvas_h, out_canvas_w, C, dtype=torch.float32, device=device)
        weights = torch.zeros(out_canvas_h, out_canvas_w, 1, dtype=torch.float32, device=device)

        for i, (x0_src, y0_src) in enumerate(positions):
            tile = upscaled_tiles[i].float()
            col  = i % cols
            row  = i // cols

            mask = torch.ones(up_h, up_w, dtype=torch.float32, device=device)

            if col > 0 and feather_px_x > 0:
                fe = min(feather_px_x, up_w)
                mask[:, :fe] *= torch.linspace(0.0, 1.0, fe, device=device)
            if col < cols - 1 and feather_px_x > 0:
                fe = min(feather_px_x, up_w)
                mask[:, up_w - fe:] *= torch.linspace(1.0, 0.0, fe, device=device)
            if row > 0 and feather_px_y > 0:
                fe = min(feather_px_y, up_h)
                mask[:fe, :] *= torch.linspace(0.0, 1.0, fe, device=device).unsqueeze(1)
            if row < rows - 1 and feather_px_y > 0:
                fe = min(feather_px_y, up_h)
                mask[up_h - fe:, :] *= torch.linspace(1.0, 0.0, fe, device=device).unsqueeze(1)

            ox = round(x0_src * scale_x)
            oy = round(y0_src * scale_y)

            dx0 = max(0, ox);          dy0 = max(0, oy)
            dx1 = min(out_canvas_w, ox + up_w)
            dy1 = min(out_canvas_h, oy + up_h)
            tx0 = dx0 - ox;            ty0 = dy0 - oy
            tx1 = tx0 + (dx1 - dx0);   ty1 = ty0 + (dy1 - dy0)

            if tx1 <= tx0 or ty1 <= ty0:
                continue

            mask_crop = mask[ty0:ty1, tx0:tx1].unsqueeze(-1)
            canvas [dy0:dy1, dx0:dx1, :]  += tile[ty0:ty1, tx0:tx1, :] * mask_crop
            weights[dy0:dy1, dx0:dx1, :]  += mask_crop

        result = (canvas / weights.clamp(min=1e-8)).clamp(0.0, 1.0).to(dtype)

        # Crop/resize back to exact original aspect ratio at the effective upscale.
        # Use geometric mean of x/y scale factors so the output area is consistent,
        # then derive final_w and final_h directly from orig_w/orig_h — this
        # guarantees the aspect ratio is pixel-perfect regardless of canvas padding.
        eff_scale = math.sqrt((out_canvas_w / orig_w) * (out_canvas_h / orig_h))
        final_w   = round(orig_w * eff_scale)
        final_h   = round(orig_h * eff_scale)

        if final_w != out_canvas_w or final_h != out_canvas_h:
            result = _resize(result, final_h, final_w)

        print(f"  Output : {final_w}x{final_h}  ({final_w/orig_w:.3f}x  {final_h/orig_h:.3f}x)")

        return (result.unsqueeze(0),)


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SeedVR2TileSplitter":   SeedVR2TileSplitter,
    "SeedVR2TileStitcher":   SeedVR2TileStitcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2TileSplitter":   "SeedVR2 Tile Splitter",
    "SeedVR2TileStitcher":   "SeedVR2 Tile Stitcher",
}
