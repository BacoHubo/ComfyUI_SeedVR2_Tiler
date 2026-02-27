# ComfyuUI SeedVR2 Tiler

<img width="512" height="512" alt="Tiler Workflow" src="https://github.com/user-attachments/assets/0753a738-03f1-45ee-8988-05e0c3fa6e18" />


A ComfyUI custom node pack for tiling large images through [SeedVR2](https://github.com/TencentARC/SeedVR) with overlap blending. Allows SeedVR2 to upscale images of any size by splitting them into tiles, processing each tile, and seamlessly stitching them back together.

## Nodes

### SeedVR2 Tile Splitter
Splits an image into a batch of overlapping tiles sized for SeedVR2's resolution constraints.

**Inputs**
- `image` — source image
- `tile_size_mp` — maximum tile size in megapixels (default 1.0). Lower = less VRAM per pass
- `tile_upscale_mp` — target resolution for SeedVR2 to upscale each tile to, in megapixels
- `overlap_fraction` — overlap between adjacent tiles as a fraction of tile size (default 0.1)
- `feather_blend` — blend width for overlap stitching (0–1)

**Outputs**
- `tiles` — IMAGE batch ready for SeedVR2
- `tile_metadata` — internal metadata needed by the Stitcher
- `resolution` — INT hint to wire into SeedVR2's resolution input

---

### SeedVR2 Tile Stitcher
Reassembles the upscaled tile batch back into a single image using feathered blending over the overlap regions. The output is resized to preserve the exact aspect ratio of the original image.

**Inputs**
- `upscaled_tiles` — IMAGE batch from SeedVR2
- `tile_metadata` — from the Splitter

**Outputs**
- `image` — final stitched image at the correct aspect ratio

---

## Workflow
```
Load Image → Tile Splitter → tiles ──────────────────→ SeedVR2
                           → tile_metadata ───────────→ Tile Stitcher
                           → resolution ─────────────→ SeedVR2
                                            SeedVR2 → Tile Stitcher → Save Image
```

<img width="1938" height="664" alt="Screenshot 2026-02-26 185301" src="https://github.com/user-attachments/assets/6c31d0b9-3c76-4a70-8a1b-2e0576d072ff" />

* Connect resolution to both resolution and max_resolution inputs on SeedVR2

For **multi-pass upscaling**, run the pipeline multiple times feeding the output back as input. Each pass progressively increases resolution.

---

## Installation

### Via ComfyUI Manager
Search for **SeedVR2 Tiler** in the ComfyUI Manager node list.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/BacoHubo/ComfyUI_SeedVR2_Tiler
```
No additional dependencies — uses only PyTorch and standard ComfyUI libraries.

---

## Requirements
- ComfyUI
- [SeedVR2](https://github.com/TencentARC/SeedVR) custom node installed separately
- PyTorch (included with ComfyUI)

---

## Notes
- `tile_size_mp` of 0.5–1.0 works well for most 8GB VRAM GPUs
- For poor quality source images, setting `tile_upscale_mp` close to `tile_size_mp` causes SeedVR2 to behave more as a restorer than an upscaler, often producing better results
- Multi-pass upscaling works well — each pass feeds back into the Splitter as the new source image

---

## Development
This node pack was developed with the assistance of [Claude](https://claude.ai) (Anthropic). The architecture, design decisions, and testing were directed by the author; the code was written collaboratively with AI. Shared here in the spirit of transparency.

---

## Acknowledgements
Inspired by tiling approaches in the ComfyUI community, including
[moonwhaler]([https://github.com/moonwhaler](https://github.com/moonwhaler/comfyui-seedvr2-tilingupscaler)) and the
[Steudio](https://civitai.com/models/982985/divide-and-conquer-ultimate-upscaling-workflow-for-comfyui) upscaling workflow.
SeedVR2 itself is by [TencentARC](https://github.com/TencentARC/SeedVR) — this node pack just makes it easier to use on larger images.
