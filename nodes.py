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


def _pad_to_canvas(img: torch.Tensor, canvas_h: int, canvas_w: int) -> torch.Tensor:
    """Pad image to canvas size with edge replication (no resize, no blur).
    Padding is added to the right and bottom only."""
    H, W, C = img.shape
    if H == canvas_h and W == canvas_w:
        return img
    # F.pad expects (N, C, H, W) and pads last dims first: (left, right, top, bottom)
    x = img.permute(2, 0, 1).unsqueeze(0)
    pad_w = canvas_w - W
    pad_h = canvas_h - H
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
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

    def _split_core(self, image, tile_size_mp, overlap_percent, feather_blend, upscale_ratio=None, tile_upscale_mp=None):
        """
        Core split logic shared by all variant nodes.
        Provide either upscale_ratio (preferred, uses actual tile dims) or tile_upscale_mp.
        """
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

        # Single tile fast path — send original image untouched, no padding or
        # canvas resize. Stitcher will return SeedVR2 output pixel-perfect.
        if cols == 1 and rows == 1:
            tiles_tensor = img.unsqueeze(0)
            positions    = [(0, 0)]
            tile_w       = orig_W
            tile_h       = orig_H
            canvas_w     = orig_W
            canvas_h     = orig_H
        else:
            # Pad to canvas instead of resizing — preserves original pixel values
            # with zero blur. Padding is replicated edge pixels, removed by stitcher.
            canvas_img = _pad_to_canvas(img, canvas_h, canvas_w)

            positions: List[Tuple[int, int]] = []
            tiles: List[torch.Tensor] = []
            for row in range(rows):
                for col in range(cols):
                    x0 = col * stride_w
                    y0 = row * stride_h
                    tiles.append(canvas_img[y0 : y0 + tile_h, x0 : x0 + tile_w, :])
                    positions.append((x0, y0))

            tiles_tensor = torch.stack(tiles, dim=0)

        # Compute resolution hint from actual tile dimensions.
        # If upscale_ratio provided, derive target tile dims directly — this is
        # accurate regardless of tile count or tile_size_mp value.
        # If tile_upscale_mp provided (legacy MP mode), use the old formula.
        if upscale_ratio is not None:
            target_tile_w = tile_w * upscale_ratio
            target_tile_h = tile_h * upscale_ratio
            resolution = _align(min(target_tile_w, target_tile_h), 8)
            # Exact target output dimensions for stitcher to hit precisely
            target_out_w = round(orig_W * upscale_ratio)
            target_out_h = round(orig_H * upscale_ratio)
        else:
            tile_aspect  = tile_w / tile_h
            out_h_raw    = math.sqrt(tile_upscale_mp * 1_000_000 / tile_aspect)
            resolution   = _align(min(out_h_raw * tile_aspect, out_h_raw), 8)
            target_out_w = None
            target_out_h = None

        meta: Dict[str, Any] = {
            "orig_w": orig_W, "orig_h": orig_H, "orig_c": C,
            "target_out_w": target_out_w, "target_out_h": target_out_h,
            "pad_w": canvas_w - orig_W, "pad_h": canvas_h - orig_H,
            "canvas_w": canvas_w, "canvas_h": canvas_h,
            "cols": cols, "rows": rows, "n_tiles": cols * rows,
            "tile_w": tile_w, "tile_h": tile_h,
            "overlap_w": overlap_w, "overlap_h": overlap_h,
            "stride_w": stride_w, "stride_h": stride_h,
            "positions": positions,
            "feather_blend": feather_blend,
            "resolution": resolution,
        }

        pad_info = f"  (padded +{canvas_w-orig_W}x{canvas_h-orig_H})" if (canvas_w != orig_W or canvas_h != orig_H) else "  (exact)"
        print(
            f"\n[SeedVR2 Tile Splitter]"
            f"\n  Original : {orig_W}x{orig_H}"
            f"\n  Canvas   : {canvas_w}x{canvas_h}{pad_info}"
            f"\n  Grid     : {cols}x{rows} = {cols*rows} tiles"
            f"\n  Tile     : {tile_w}x{tile_h} = {tile_w*tile_h/1e6:.2f} MP"
            f"\n  Overlap  : {overlap_w}x{overlap_h} px  |  Stride: {stride_w}x{stride_h} px"
            f"\n  SeedVR2 resolution hint: {resolution}"
        )

        return (tiles_tensor, meta, resolution)

    def split(self, image, tile_size_mp, tile_upscale_mp, overlap_percent, feather_blend):
        return self._split_core(image, tile_size_mp, overlap_percent, feather_blend,
                                tile_upscale_mp=tile_upscale_mp)


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

        pad_w = meta.get("pad_w", 0)
        pad_h = meta.get("pad_h", 0)

        target_out_w = meta.get("target_out_w")
        target_out_h = meta.get("target_out_h")

        # Fast path: single tile, no padding, no target size — return SeedVR2
        # output completely untouched before any canvas/blending math runs.
        if n_tiles == 1 and pad_w == 0 and pad_h == 0 and not target_out_w:
            final_w = up_w
            final_h = up_h
            print(f"  Output : {final_w}x{final_h}  ({final_w/orig_w:.3f}x  {final_h/orig_h:.3f}x)  [passthrough]")
            return (upscaled_tiles,)

        # Single tile path — crop padding if any, then resize to exact target if set
        if n_tiles == 1:
            tile = upscaled_tiles[0]
            if pad_w > 0 or pad_h > 0:
                crop_w = up_w - round(pad_w * scale_x)
                crop_h = up_h - round(pad_h * scale_y)
                tile = tile[:crop_h, :crop_w, :]
            if target_out_w and target_out_h:
                if tile.shape[1] != target_out_w or tile.shape[0] != target_out_h:
                    tile = _resize(tile, target_out_h, target_out_w)
            final_w = tile.shape[1]
            final_h = tile.shape[0]
            print(f"  Output : {final_w}x{final_h}  ({final_w/orig_w:.3f}x  {final_h/orig_h:.3f}x)  [single tile]")
            return (tile.unsqueeze(0),)

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

        # Crop upscaled padding before final resize
        if pad_w > 0 or pad_h > 0:
            scale_x_actual = out_canvas_w / canvas_w
            scale_y_actual = out_canvas_h / canvas_h
            crop_w = out_canvas_w - round(pad_w * scale_x_actual)
            crop_h = out_canvas_h - round(pad_h * scale_y_actual)
            result = result[:crop_h, :crop_w, :]

        target_out_w = meta.get("target_out_w")
        target_out_h = meta.get("target_out_h")

        if target_out_w and target_out_h:
            final_w, final_h = target_out_w, target_out_h
        else:
            eff_scale = math.sqrt((result.shape[1] / orig_w) * (result.shape[0] / orig_h))
            final_w   = round(orig_w * eff_scale)
            final_h   = round(orig_h * eff_scale)

        if result.shape[1] != final_w or result.shape[0] != final_h:
            result = _resize(result, final_h, final_w)

        print(f"  Output : {final_w}x{final_h}  ({final_w/orig_w:.3f}x  {final_h/orig_h:.3f}x)")

        return (result.unsqueeze(0),)


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 – Tile Splitter (by Longest Edge)
# ─────────────────────────────────────────────────────────────────────────────

class SeedVR2TileSplitterByLongestEdge(SeedVR2TileSplitter):
    """
    Same as Tile Splitter but lets you specify the desired output size as a
    maximum longest edge in pixels rather than megapixels.
    tile_upscale_mp is computed automatically from the image aspect ratio.
    """

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
                        "tooltip": "Maximum size of each tile in megapixels. Lower = less VRAM per pass.",
                    },
                ),
                "longest_edge_px": (
                    "INT",
                    {
                        "default": 2048, "min": 64, "max": 16384, "step": 8,
                        "display": "number",
                        "tooltip": "Desired pixel length of the longest edge in the final stitched output.",
                    },
                ),
                "overlap_percent": (
                    "FLOAT",
                    {
                        "default": 10.0, "min": 0.0, "max": 40.0, "step": 1.0,
                        "display": "number",
                        "tooltip": "Overlap between adjacent tiles as a % of tile size.",
                    },
                ),
                "feather_blend": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                        "display": "number",
                        "tooltip": "1.0 = smooth feather blend. 0.0 = hard cut.",
                    },
                ),
            }
        }

    def split(self, image, tile_size_mp, longest_edge_px, overlap_percent, feather_blend):
        img = image[0] if image.ndim == 4 else image
        orig_H, orig_W = img.shape[:2]
        upscale_ratio  = longest_edge_px / max(orig_W, orig_H)
        return self._split_core(image, tile_size_mp, overlap_percent, feather_blend,
                                upscale_ratio=upscale_ratio)


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 – Tile Splitter (by Shortest Edge)
# ─────────────────────────────────────────────────────────────────────────────

class SeedVR2TileSplitterByShortestEdge(SeedVR2TileSplitter):
    """
    Same as Tile Splitter but lets you specify the desired output size as a
    minimum shortest edge in pixels rather than megapixels.
    tile_upscale_mp is computed automatically from the image aspect ratio.
    """

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
                        "tooltip": "Maximum size of each tile in megapixels. Lower = less VRAM per pass.",
                    },
                ),
                "shortest_edge_px": (
                    "INT",
                    {
                        "default": 2048, "min": 64, "max": 16384, "step": 8,
                        "display": "number",
                        "tooltip": "Desired pixel length of the shortest edge in the final stitched output.",
                    },
                ),
                "overlap_percent": (
                    "FLOAT",
                    {
                        "default": 10.0, "min": 0.0, "max": 40.0, "step": 1.0,
                        "display": "number",
                        "tooltip": "Overlap between adjacent tiles as a % of tile size.",
                    },
                ),
                "feather_blend": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                        "display": "number",
                        "tooltip": "1.0 = smooth feather blend. 0.0 = hard cut.",
                    },
                ),
            }
        }

    def split(self, image, tile_size_mp, shortest_edge_px, overlap_percent, feather_blend):
        img = image[0] if image.ndim == 4 else image
        orig_H, orig_W = img.shape[:2]
        upscale_ratio  = shortest_edge_px / min(orig_W, orig_H)
        return self._split_core(image, tile_size_mp, overlap_percent, feather_blend,
                                upscale_ratio=upscale_ratio)


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 – Tile Splitter (by Upscale Factor)
# ─────────────────────────────────────────────────────────────────────────────

class SeedVR2TileSplitterByFactor(SeedVR2TileSplitter):
    """
    Same as Tile Splitter but lets you specify the desired upscale as a
    simple multiplier (e.g. 2.0 = double the image dimensions).
    tile_upscale_mp is computed automatically.
    """

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
                        "tooltip": "Maximum size of each tile in megapixels. Lower = less VRAM per pass.",
                    },
                ),
                "upscale_factor": (
                    "FLOAT",
                    {
                        "default": 2.0, "min": 1.0, "max": 16.0, "step": 0.25,
                        "display": "number",
                        "tooltip": "Upscale multiplier. 2.0 = double the width and height of the final output.",
                    },
                ),
                "overlap_percent": (
                    "FLOAT",
                    {
                        "default": 10.0, "min": 0.0, "max": 40.0, "step": 1.0,
                        "display": "number",
                        "tooltip": "Overlap between adjacent tiles as a % of tile size.",
                    },
                ),
                "feather_blend": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                        "display": "number",
                        "tooltip": "1.0 = smooth feather blend. 0.0 = hard cut.",
                    },
                ),
            }
        }

    def split(self, image, tile_size_mp, upscale_factor, overlap_percent, feather_blend):
        return self._split_core(image, tile_size_mp, overlap_percent, feather_blend,
                                upscale_ratio=upscale_factor)


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SeedVR2TileSplitter":                 SeedVR2TileSplitter,
    "SeedVR2TileStitcher":                 SeedVR2TileStitcher,
    "SeedVR2TileSplitterByLongestEdge":    SeedVR2TileSplitterByLongestEdge,
    "SeedVR2TileSplitterByShortestEdge":   SeedVR2TileSplitterByShortestEdge,
    "SeedVR2TileSplitterByFactor":         SeedVR2TileSplitterByFactor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2TileSplitter":                 "SeedVR2 Tile Splitter",
    "SeedVR2TileStitcher":                 "SeedVR2 Tile Stitcher",
    "SeedVR2TileSplitterByLongestEdge":    "SeedVR2 Tile Splitter (Longest Edge)",
    "SeedVR2TileSplitterByShortestEdge":   "SeedVR2 Tile Splitter (Shortest Edge)",
    "SeedVR2TileSplitterByFactor":         "SeedVR2 Tile Splitter (Upscale Factor)",
}
