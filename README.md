# LightGenBench: A Benchmark for 3D Emission Generation

![LightGenBench assets with their emission on](docs/assets/hero_on.jpg)

[Dongchen Yang](https://www.sfu.ca/~dya78/), [Xingguang Yan](https://yanxg.art/), [Manolis Savva](https://msavva.github.io/) \
Simon Fraser University

[![Project Page](https://img.shields.io/badge/Project%20Page-1f1f1f?style=for-the-badge&logo=data:image%2Fsvg%2Bxml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHZpZXdCb3g9JzAgMCAzMiAzMic%2BPGNpcmNsZSBjeD0nMTYnIGN5PScxNicgcj0nNycgZmlsbD0nI2Y1OWUwYicvPjxjaXJjbGUgY3g9JzE2JyBjeT0nMTYnIHI9JzEzJyBmaWxsPSdub25lJyBzdHJva2U9JyNmNTllMGInIHN0cm9rZS1vcGFjaXR5PScuNDUnIHN0cm9rZS13aWR0aD0nMi41Jy8%2BPC9zdmc%2B)](https://3dlg-hcvc.github.io/lightgenbench/)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-lightgrey?style=for-the-badge&logo=arxiv&logoColor=white)](#)
[![Dataset](https://img.shields.io/badge/Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/3dlg-hcvc/LightgenBench)

LightGenBench is a benchmark for emission generation.
We curate a [dataset](#dataset) of 36,826 artist-made emissive assets in 105 categories from [TexVerse](https://huggingface.co/datasets/YiboZhang2001/TexVerse), and evaluate [baselines](#baselines) from four method families: UV-domain, sparse-voxel, multi-view, and segmentation-based generation.


## Environment Setup
Coming soon.


## Dataset
Submit the short access form on the [Hugging Face dataset page](https://huggingface.co/datasets/3dlg-hcvc/LightgenBench) once (access is granted right away), then log in and download (about 117 GB):
```bash
hf auth login
hf download 3dlg-hcvc/LightgenBench --repo-type dataset --local-dir lightgenbench
cd lightgenbench && for t in data/*/*/*.tar; do tar -xf "$t"; done
```
The dataset is split into 36,426 train, 200 validation and 200 test shapes.
See the [dataset card](https://huggingface.co/datasets/3dlg-hcvc/LightgenBench) for the file layout and data format.


## Baselines
For the UV domain, we use TEXGen-Emission; for the voxel domain, TRELLIS.2-Emission; for the multi-view domain, Hunyuan3D-Emission.
We also include a segmentation-based approach, SegviGen-Emission, which works on voxels.
Code and checkpoints: coming soon.


## Evaluation
The validation split is used for model selection and the test set for the reported numbers.
We sample 50,000 points on each asset's surface and report IoU for the emissive mask, and MAE and PSNR on the emissive color, averaged over five random seeds.
Code: coming soon.
<!-- TODO(code release): the evaluation samples points on the TexVerse GLBs, which the dataset does not include; say how to get them. -->


## Citation
```
@inproceedings{lightgenbench2026,
  title  = {LightGenBench: A Benchmark for 3D Emission Generation},
  author = {Yang, Dongchen and Yan, Xingguang and Savva, Manolis},
  year   = {2026}
}
```
LightGenBench is built on [TexVerse](https://huggingface.co/datasets/YiboZhang2001/TexVerse), so please cite it as well.
Every shape keeps the license of its source Sketchfab model; `metadata.parquet` lists the license, author and source URL of each.


## Acknowledgements
To be added.
