# ComfyUI SeedVR2 Tiler

<img width="512" height="512" alt="SeedVR2 Tiler Workflow" src="https://github.com/user-attachments/assets/5987e2ce-6885-4618-9aa0-7b1382cdd0aa" />


A ComfyUI custom node pack for tiling large images through [SeedVR2](https://github.com/TencentARC/SeedVR) with overlap blending. Allows SeedVR2 to upscale images of any size by splitting them into tiles, processing each tile, and seamlessly stitching them back together.

## Updates

**v1.2.0**
- Fixed single tile passthrough — output is now identical to running SeedVR2 directly
- Switched from canvas resize to edge-replication padding, preserving original pixel values

**v1.1.0**
- Added Longest Edge, Shortest Edge, and Upscale Factor splitter nodes

**v1.0.0**
- Initial release with Tile Splitter and Tile Stitcher nodes

## Nodes

### SeedVR2 Tile Splitter
Splits an image into a batch of overlapping tiles sized for SeedVR2's resolution constraints. When the image fits in a single tile, it is passed through completely untouched — output is identical to running SeedVR2 directly.

**Inputs**
- `image` — source image
- `tile_size_mp` — maximum tile size in megapixels (default 1.0). Lower = less VRAM per pass
- `tile_upscale_mp` — target resolution for SeedVR2 to upscale each tile to, in megapixels
- `overlap_percent` — overlap between adjacent tiles as a percentage (default 10)
- `feather_blend` — blend width for overlap stitching (0–1)

**Outputs**
- `tiles` — IMAGE batch ready for SeedVR2
- `tile_metadata` — internal metadata needed by the Stitcher
- `resolution` — INT hint to wire into SeedVR2's resolution input

---

### SeedVR2 Tile Splitter (Longest Edge)
Same as the standard Tile Splitter but lets you specify your desired output size as a longest edge in pixels. The Stitcher will resize the final output to hit the exact target dimensions.

**Inputs**
- `image` — source image
- `tile_size_mp` — maximum tile size in megapixels. Lower = less VRAM per pass
- `longest_edge_px` — desired pixel length of the longest edge in the final stitched output
- `overlap_percent` — overlap between adjacent tiles as a percentage (default 10)
- `feather_blend` — blend width for overlap stitching (0–1)

---

### SeedVR2 Tile Splitter (Shortest Edge)
Same as the standard Tile Splitter but lets you specify your desired output size as a shortest edge in pixels. The Stitcher will resize the final output to hit the exact target dimensions.

**Inputs**
- `image` — source image
- `tile_size_mp` — maximum tile size in megapixels. Lower = less VRAM per pass
- `shortest_edge_px` — desired pixel length of the shortest edge in the final stitched output
- `overlap_percent` — overlap between adjacent tiles as a percentage (default 10)
- `feather_blend` — blend width for overlap stitching (0–1)

---

### SeedVR2 Tile Splitter (Upscale Factor)
Same as the standard Tile Splitter but lets you specify a simple upscale multiplier instead of megapixels. The Stitcher will resize the final output to hit the exact target dimensions.

**Inputs**
- `image` — source image
- `tile_size_mp` — maximum tile size in megapixels. Lower = less VRAM per pass
- `upscale_factor` — upscale multiplier (e.g. 2.0 = 2× output, 1.5 = 1.5× output)
- `overlap_percent` — overlap between adjacent tiles as a percentage (default 10)
- `feather_blend` — blend width for overlap stitching (0–1)

---

### SeedVR2 Tile Stitcher
Reassembles the upscaled tile batch back into a single image using feathered blending over the overlap regions.

**Inputs**
- `upscaled_tiles` — IMAGE batch from SeedVR2
- `tile_metadata` — from the Splitter

**Outputs**
- `image` — final stitched image

---

## Workflow
```
Load Image → Tile Splitter → tiles ──────────────────→ SeedVR2
                           → tile_metadata ───────────→ Tile Stitcher
                           → resolution ─────────────→ SeedVR2
                                            SeedVR2 → Tile Stitcher → Save Image
```

<img width="2240" height="954" alt="Workflow" src="https://github.com/user-attachments/assets/6b3265bf-8230-4b26-85b0-9075b2a77e12" />


- Connect `resolution` to SeedVR2's `resolution` input. Set `max_resolution` to 0 to disable the longest edge cap.

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
- The Longest Edge, Shortest Edge, and Upscale Factor nodes apply a final resize to hit exact output dimensions. The base MP node does not — it returns whatever SeedVR2 produces
- Multi-pass upscaling works well — each pass feeds back into the Splitter as the new source image

---

## Support
If you find this useful, consider buying me a coffee!

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/dbacon/tip)

---

## Development
This node pack was developed with the assistance of [Claude](https://claude.ai) (Anthropic). The architecture, design decisions, and testing were directed by the author; the code was written collaboratively with AI. Shared here in the spirit of transparency.

---

## Acknowledgements
Inspired by tiling approaches in the ComfyUI community, including
[Moonwhaler](https://github.com/moonwhaler/comfyui-seedvr2-tilingupscaler) and the
[Steudio](https://civitai.com/models/982985/divide-and-conquer-ultimate-upscaling-workflow-for-comfyui) upscaling workflow.
SeedVR2 itself is by [TencentARC](https://github.com/TencentARC/SeedVR) — this node pack just makes it easier to use on larger images.
