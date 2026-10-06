# LightGenBench: A Benchmark for 3D Emission Generation

![LightGenBench shapes with their emission on](docs/assets/hero_on.jpg)

[Dongchen Yang](https://www.sfu.ca/~dya78/), [Xingguang Yan](https://yanxg.art/), [Manolis Savva](https://msavva.github.io/) \
Simon Fraser University

[![Project Page](https://img.shields.io/badge/Project%20Page-1f1f1f?style=for-the-badge&logo=data:image%2Fsvg%2Bxml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHZpZXdCb3g9JzAgMCAzMiAzMic%2BPGNpcmNsZSBjeD0nMTYnIGN5PScxNicgcj0nNycgZmlsbD0nI2Y1OWUwYicvPjxjaXJjbGUgY3g9JzE2JyBjeT0nMTYnIHI9JzEzJyBmaWxsPSdub25lJyBzdHJva2U9JyNmNTllMGInIHN0cm9rZS1vcGFjaXR5PScuNDUnIHN0cm9rZS13aWR0aD0nMi41Jy8%2BPC9zdmc%2B)](https://3dlg-hcvc.github.io/lightgenbench/)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-lightgrey?style=for-the-badge&logo=arxiv&logoColor=white)](#)
[![Dataset](https://img.shields.io/badge/Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/3dlg-hcvc/LightgenBench)

LightGenBench is a dataset of emissive 3D objects for training and evaluating emission texture generation.
It holds 36,826 shapes from [TexVerse](https://huggingface.co/datasets/YiboZhang2001/TexVerse), each with albedo, metallic, roughness, opacity and an emission map, stored as a UV atlas, sparse voxels and six rendered views.

> [!NOTE]
> The code (environment, data loader, baselines and evaluation) is coming soon; the sections marked *Coming soon* will be filled in then.
> `docs/` holds the project page, served by GitHub Pages.



## Environment Setup
*Coming soon.*
<!-- TODO(code release): conda environment file and install commands. -->



## Dataset Setup

### 1. Download LightGenBench
The dataset is on [Hugging Face](https://huggingface.co/datasets/3dlg-hcvc/LightgenBench) and needs no access request.
Install the Hugging Face command-line tools and download it (about 117 GB):
```bash
python -m pip install -U huggingface_hub

# Everything
hf download 3dlg-hcvc/LightgenBench --repo-type dataset --local-dir /path/to/lightgenbench

# Or the root files plus one representation for every split (here the voxels)
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
│   ├── atlas.npz            # 512×512 UV atlas: position, normal, albedo, metallic, roughness, opacity, emission
│   ├── emission_voxels.vxz  # 256³ sparse voxels: emission
│   ├── pbr_voxels.vxz       # the same voxels: albedo, metallic, roughness, opacity
│   ├── multiview            # six orthographic 512×512 views, six maps each, and transforms.json
│   └── thumbnail.png        # the Sketchfab preview image
├── ...
```

**Storage Requirements**
- Train: 36,426 shapes, ~115 GB
- Val: 200 shapes, ~0.6 GB
- Test: 200 shapes, ~0.7 GB

By representation: UV atlas ~69 GB, multiview ~32 GB, voxels ~10 GB, thumbnails ~6 GB.
Unpacking needs about the same space again until you delete `data/`.


### 3. Download the TexVerse Meshes
*Coming soon.*
The evaluation samples points on the source mesh of each shape, which LightGenBench does not include.
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

# Six orthographic views, each with six maps (albedo, mr, normal, pos, emission, alpha)
view = np.asarray(Image.open(f"{root}/{uuid}/multiview/000_emission.png"))
cameras = json.load(open(f"{root}/{uuid}/multiview/transforms.json"))
```
The voxel files are read with the `o_voxel` package of [TRELLIS.2](https://github.com/microsoft/TRELLIS.2):
```python
import o_voxel

coords, attrs = o_voxel.io.read_vxz(f"{root}/{uuid}/emission_voxels.vxz")
# coords: int32, N×3 grid indices (0–255); attrs["emissive"]: uint8, N×3, linear RGB
```
A data loader for training and evaluation: *Coming soon.*

**Refer to the [dataset card](https://huggingface.co/datasets/3dlg-hcvc/LightgenBench) for more information on the data format.**



## Baselines
We adapt four 3D generators to emission generation, covering all three representations:

| Baseline | Representation | Built on |
|---|---|---|
| TEXGen-Emission | UV atlas | [TEXGen](https://github.com/CVMI-Lab/TEXGen) |
| TRELLIS.2-Emission | sparse voxels | [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) |
| SegviGen-Emission | sparse voxels | [SegviGen](https://github.com/Nelipot-Lee/SegviGen) |
| Hunyuan3D-Emission | six views | [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) |

Training and inference code and checkpoints: *Coming soon.*
<!-- TODO(code release): per-baseline training and inference commands, checkpoint download. -->



## Evaluation Protocol
Every method gets the same input for a shape: its geometry, its non-emissive material maps, and its thumbnail image. It predicts the emission.
The validation split is used to pick checkpoints; the test split (200 shapes) produces the reported numbers.

For each test shape, we sample 50,000 points uniformly on the mesh and read the predicted emission at each point from the method's own output by a nearest-neighbor lookup (the nearest occupied voxel, or the nearest atlas texel).
We report IoU of the emissive points (prediction and reference both thresholded at 1/255), and MAE and PSNR of the emission color over all sampled points.
Generative baselines are run with five random seeds, and we report the mean.

Evaluation code: *Coming soon.*
<!-- TODO(code release): the evaluation command and its output format. -->



## Citation
If you find LightGenBench helpful in your research, please cite our work:
<!-- TODO: venue and pages once the paper is public. -->
```
@inproceedings{lightgenbench2026,
  title  = {{LightGenBench}: A Benchmark for {3D} Emission Generation},
  author = {Yang, Dongchen and Yan, Xingguang and Savva, Manolis},
  year   = {2026}
}
```
LightGenBench is built on [TexVerse](https://huggingface.co/datasets/YiboZhang2001/TexVerse), so please cite it as well.



## License
Every shape keeps the license of its source model on Sketchfab; `metadata.parquet` lists the license, author and source URL of each shape, and the [dataset card](https://huggingface.co/datasets/3dlg-hcvc/LightgenBench#license) explains what each license allows.
The code license: *Coming soon.*



## Acknowledgements
*Coming soon.*
<!-- TODO: funding and thanks, as in the paper's camera-ready version. -->
