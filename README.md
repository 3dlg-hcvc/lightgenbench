# LightGenBench: A Benchmark for 3D Emission Generation

![LightGenBench assets with their emission on](docs/assets/hero_on.jpg)

[Dongchen Yang](https://www.sfu.ca/~dya78/), [Xingguang Yan](https://yanxg.art/), [Manolis Savva](https://msavva.github.io/) \
Simon Fraser University

[![Project Page](https://img.shields.io/badge/Project%20Page-1f1f1f?style=for-the-badge&logo=data:image%2Fsvg%2Bxml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHZpZXdCb3g9JzAgMCAzMiAzMic%2BPGNpcmNsZSBjeD0nMTYnIGN5PScxNicgcj0nNycgZmlsbD0nI2Y1OWUwYicvPjxjaXJjbGUgY3g9JzE2JyBjeT0nMTYnIHI9JzEzJyBmaWxsPSdub25lJyBzdHJva2U9JyNmNTllMGInIHN0cm9rZS1vcGFjaXR5PScuNDUnIHN0cm9rZS13aWR0aD0nMi41Jy8%2BPC9zdmc%2B)](https://3dlg-hcvc.github.io/lightgenbench/)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-lightgrey?style=for-the-badge&logo=arxiv&logoColor=white)](#)
[![Dataset](https://img.shields.io/badge/Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/3dlg-hcvc/LightgenBench)

LightGenBench is a benchmark for emission generation: predicting a spatially varying RGB emission map over the surface of a 3D asset from its geometry, non-emissive materials, and a rendered view.
It holds 36,826 artist-made emissive assets in 105 categories, curated from [TexVerse](https://huggingface.co/datasets/YiboZhang2001/TexVerse), each with albedo, metallic, roughness, opacity and emission.
Every asset comes in the three data representations of the UV, voxel and multi-view domains: UV atlases, O-Voxels and multi-view images.

> [!NOTE]
> The code (environment, data loader, baselines and evaluation) is coming soon; the sections marked *Coming soon* will be filled in then.
> `docs/` holds the project page, served by GitHub Pages.



## Environment Setup
*Coming soon.*
<!-- TODO(code release): conda environment file and install commands. -->



## Dataset Setup

### 1. Download LightGenBench
First, submit the short access form on the [Hugging Face dataset page](https://huggingface.co/datasets/3dlg-hcvc/LightgenBench) once (access is granted right away).
Then log in to Hugging Face on your machine ([guide](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)) and download the dataset (about 117 GB):
```bash
# Install the Hugging Face command-line tools
python -m pip install -U huggingface_hub

hf auth login

# Everything
hf download 3dlg-hcvc/LightgenBench --repo-type dataset --local-dir /path/to/lightgenbench

# Or the root files plus one representation for every split (here the O-Voxels)
hf download 3dlg-hcvc/LightgenBench splits.json metadata.parquet checksums.sha256 --repo-type dataset --local-dir /path/to/lightgenbench
hf download 3dlg-hcvc/LightgenBench --repo-type dataset --include "data/*/voxels/*" --local-dir /path/to/lightgenbench
```


### 2. Verify and Unpack
The download is a set of tars under `data/`.
Check them against `checksums.sha256`, then unpack every tar in place:
```bash
cd /path/to/lightgenbench
sha256sum -c --ignore-missing checksums.sha256
for t in data/*/*/*.tar; do tar -xf "$t"; done   # then rm -r data/ to free the space
```
After unpacking, every shape is one directory named by its TexVerse uuid:
```
lightgenbench
├── splits.json              # {"train": [uuid, ...], "val": [...], "test": [...]}
├── metadata.parquet         # one row per shape: split, category, license, author, source URL
├── 030adbdafd824eb7b6cc85b5a77f6f52
│   ├── atlas.npz            # UV atlas, 512×512: position, normal, albedo, metallic, roughness, opacity, emission
│   ├── emission_voxels.vxz  # O-Voxels on a 256³ grid: emission
│   ├── pbr_voxels.vxz       # the same voxels: albedo, metallic, roughness, opacity
│   ├── multiview            # multi-view images: six orthographic 512×512 views, six maps each, and transforms.json
│   └── thumbnail.png        # the asset's Sketchfab thumbnail
├── ...
```

**Storage Requirements**
- Train: 36,426 shapes, ~115 GB
- Validation: 200 shapes, ~0.6 GB
- Test: 200 shapes, ~0.7 GB

By representation: UV atlases ~69 GB, multi-view images ~32 GB, O-Voxels ~10 GB, thumbnails ~6 GB.
Unpacking needs about the same space again until you delete `data/`.


### 3. Download the TexVerse Meshes
*Coming soon.*
The evaluation samples points on the original asset's surface, which LightGenBench does not include.
This step will download the TexVerse GLB files of the validation and test shapes.
<!-- TODO(code release): the download command and where the evaluation expects the files. -->



## Quick Start
Load one test shape with NumPy and Pillow:
```python
import json

import numpy as np
from PIL import Image

root = "/path/to/lightgenbench"
uuid = json.load(open(f"{root}/splits.json"))["test"][0]

# UV atlas: eight 512x512 maps over one UV layout
atlas = np.load(f"{root}/{uuid}/atlas.npz")
emission = atlas["emission_color"]     # uint8, 512x512x3, linear RGB
covered = atlas["occupancy"][..., 0]   # texels the UV layout covers

# Multi-view images: six orthographic views, each with six maps (albedo, mr, normal, pos, emission, alpha)
view = np.asarray(Image.open(f"{root}/{uuid}/multiview/000_emission.png"))
cameras = json.load(open(f"{root}/{uuid}/multiview/transforms.json"))
```
The O-Voxels are read with the `o_voxel` package of [TRELLIS.2](https://github.com/microsoft/TRELLIS.2):
```python
import o_voxel

coords, attrs = o_voxel.io.read_vxz(f"{root}/{uuid}/emission_voxels.vxz")
# coords: int32, N×3 grid indices (0–255); attrs["emissive"]: uint8, N×3, linear RGB
```
A data loader for training and evaluation: *Coming soon.*

**Refer to the [dataset card](https://huggingface.co/datasets/3dlg-hcvc/LightgenBench) for more information on the data format.**



## Baselines
For the UV domain, we use TEXGen-Emission; for the voxel domain, TRELLIS.2-Emission; for the multi-view domain, Hunyuan3D-Emission.
Since one key challenge for emission generation is to localize the emissive regions, we also include a segmentation-based approach, SegviGen-Emission, which works on voxels.

| Baseline | Domain | Built on |
|---|---|---|
| TEXGen-Emission | UV | [TEXGen](https://github.com/CVMI-Lab/TEXGen) |
| TRELLIS.2-Emission | voxel | [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) |
| Hunyuan3D-Emission | multi-view | [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) |
| SegviGen-Emission | voxel (segmentation-based) | [SegviGen](https://github.com/Nelipot-Lee/SegviGen) |

Training and inference code and checkpoints: *Coming soon.*
<!-- TODO(code release): per-baseline training and inference commands, checkpoint download. -->



## Evaluation Protocol
All four baselines are trained on the same training split and receive the same input: the geometry, the non-emissive material parameters, and one thumbnail rendering of the asset (its platform preview, rendered with artist-chosen settings).
The validation split is used for model selection and the test set for the reported numbers.

For each asset, we sample 50,000 points uniformly on the input mesh.
The reference emission is the emission color of each sampled point on the original asset.
The predicted emission at each point is read from each method's own output by a nearest-neighbor lookup: the nearest occupied voxel for TRELLIS.2-Emission and SegviGen-Emission, and the nearest atlas texel by 3D position for TEXGen-Emission.
For Hunyuan3D-Emission, we look up the nearest atlas texel, since its six generated views are back-projected into a UV atlas first.

We evaluate the predictions with intersection over union (IoU) for the emissive mask, and mean absolute error (MAE) and peak signal-to-noise ratio (PSNR) on the emissive color.
IoU measures whether the emissive region is generated at the right location: we threshold both the prediction and the reference at 1/255 and report the IoU of the two point sets.
MAE and PSNR measure whether the emitted color is correct; we compute them over all sampled points.
Since a single prediction for generative baselines is an unstable measurement, we evaluate the baselines five times with different random seeds and report the mean across them.

Evaluation code: *Coming soon.*
<!-- TODO(code release): the evaluation command and its output format. -->



## Citation
If you find LightGenBench helpful in your research, please cite our work:
<!-- TODO: venue and pages once the paper is public. -->
```
@inproceedings{lightgenbench2026,
  title  = {LightGenBench: A Benchmark for 3D Emission Generation},
  author = {Yang, Dongchen and Yan, Xingguang and Savva, Manolis},
  year   = {2026}
}
```
LightGenBench is built on [TexVerse](https://huggingface.co/datasets/YiboZhang2001/TexVerse), so please cite it as well.



## License
Every shape keeps the license of its source Sketchfab model; `metadata.parquet` lists the license, author and source URL of each.
The [dataset card](https://huggingface.co/datasets/3dlg-hcvc/LightgenBench#license) explains what each license allows.
The code license: *Coming soon.*



## Acknowledgements
To be added.
<!-- TODO: funding and thanks, as in the paper's camera-ready version. -->
