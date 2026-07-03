# Seen2Scene: Completing Realistic 3D Scenes with Visibility-Guided Flow

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://quan-meng.github.io/projects/seen2scene/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2603.28548)
[![YouTube](https://img.shields.io/badge/YouTube-Video-FF0000?logo=youtube)](https://www.youtube.com/watch?v=5qJYLjMsJe8)

</div>

<p align="center">
  <img src="assets/teaser.png?v=2" alt="Seen2Scene Teaser" width="100%">
</p>

## Abstract

We present **Seen2Scene**, the first flow matching-based approach that trains directly on incomplete, real-world 3D scans for scene completion and generation. Unlike prior methods that rely on complete and hence synthetic 3D data, our approach introduces **visibility-guided flow matching**, which explicitly masks out unknown regions in real scans, enabling effective learning from real-world, partial observations. We represent 3D scenes using truncated signed distance fields (TSDFs) encoded in sparse grids and employ a sparse transformer to efficiently model complex scene structures while masking unknown regions. We employ 3D layout boxes as an input conditioning signal, and our approach is flexibly adapted to various other inputs such as text or partial scans. By learning directly from real-world, incomplete 3D scans, Seen2Scene enables realistic 3D scene completion for complex, cluttered real environments. Experiments demonstrate that our model produces coherent, complete, and realistic 3D scenes, outperforming baselines in completion accuracy and generation quality.

## Install

Create a new conda environment from the provided export. Use `--name` so conda creates a separate environment instead of modifying the currently active one:

```bash
conda env create --name seen2scene --file env/sg.yml
conda activate seen2scene

# FlashAttention: sparse/window/full attention kernels in seen2scene/models/sparse/attention/.
bash env/install_flash_attn_2_7_4.sh

# fVDB: sparse VDB tensors, fields, VAE, generator, and ControlNet model code.
bash env/install_fvdb_0_2_1.sh

# TorchSparse: sparse convolution backend in seen2scene/models/sparse/.
bash env/install_torchsparse_2_1_0.sh
```

The FlashAttention, `fvdb`, and TorchSparse scripts install into the activated `seen2scene` environment and build native/CUDA extensions, so run them on a Linux machine with `nvcc` available. The exported environment uses Python 3.11, PyTorch 2.4.1, and CUDA 12.0.

Run commands from the repository root. The examples below use `python -m ...` so Python imports this checkout without requiring package installation. For data export and rendering, also install Blender/BlenderProc and set dataset paths with the `SEEN2SCENE_*_DIR` environment variables or in `seen2scene/configs/dataset.py`.

## Data Processing

Set dataset paths with the `SEEN2SCENE_*_DIR` environment variables or in `seen2scene/configs/dataset.py`, then run the exporters:

ARKitScenes and ScanNet++ raw fusion use VDBFusion. If you need to generate fusion files from raw scans, install VDBFusion separately or adapt `env/install_vdbfusion_0_1_6.sh` for your system OpenVDB/CMake setup. This step is not required when `fusion_p_*.vdb` files are already prepared.

```bash
# 3D-FRONT
# input: 3D-FRONT JSON layouts, 3D-FUTURE models, textures
# exports: scene meshes, cameras, and metadata
python -m seen2scene.data.export_front3d export_data

# input: exported meshes and cameras
# exports: rendered RGB-D or depth scans
python -m seen2scene.data.export_front3d export_scans

# input: rendered depth scans
# exports: fused TSDF/VDB scenes
python -m seen2scene.data.export_front3d export_fusion

# ARKitScenes
python -m seen2scene.data.export_arkitscenes download_1mca
python -m seen2scene.data.export_arkitscenes export_fusion
python -m seen2scene.data.export_arkitscenes export_data

# ScanNet++
# TODO: ScanNet++ fusion export requires raw LiDAR data
# python -m seen2scene.data.export_scannetpp export_fusion
python -m seen2scene.data.export_scannetpp export_data
```

ScanNet++ export expects a scene mapping CSV; set the mapping path in the ScanNet++ data config before running it.

`submit_jobs` can also run locally; set `cluster="local"` in the `Slurm` config or run on a CUDA machine where local execution is selected automatically.

## Training

Training uses `python -m seen2scene.main` with `tyro` configs. Run commands from the repository root with the `seen2scene` conda environment activated.

The examples below train on all prepared datasets. If you only prepared a subset, change `--data.data-list` accordingly, for example `--data.data-list 3D-FRONT`.

Prepared data is expected to contain scene folders under each configured dataset root. For these commands, each scene needs `meta.json` and `fusion_p_1.0_v_0.011.vdb`; completion training also needs `fusion_p_0.1_v_0.011.vdb`. Validation metrics and visualization use `scene.ply` when available.

```bash
# Train VAE
python -m seen2scene.main vae \
  --slurm.cluster local \
  --strategy auto \
  --batch-size 2 \
  --latent-key tsdf_p_1.0 \
  --data.data-list 3D-FRONT ScannetPP ARKitScenes

# Train Flow Matching generator.
# AE_LOG is the VAE experiment folder under experiments/auto_encoder.
python -m seen2scene.main generator \
  --ae-log AE_LOG \
  --slurm.cluster local \
  --strategy auto \
  --precision bf16-mixed \
  --batch-size 16 \
  --latent-key tsdf_p_1.0 \
  --progressive-end 250 \
  --data.data-list 3D-FRONT ScannetPP ARKitScenes

# Train ControlNet.
# GEN_LOG is the Flow Matching experiment folder under experiments/auto_encoder/AE_LOG/generator.
python -m seen2scene.main control \
  --ae-log AE_LOG \
  --gen-log GEN_LOG \
  --slurm.cluster local \
  --strategy auto \
  --precision bf16-mixed \
  --batch-size 1 \
  --num-sanity-val-steps 1 \
  --progressive-end 30 \
  --max-epochs 60 \
  --src-key tsdf_p_0.1 \
  --latent-key tsdf_p_1.0 \
  --data.data-list 3D-FRONT ScannetPP ARKitScenes
```

Check available options with `python -m seen2scene.main {vae,generator,control} --help`.

## Checkpoints

Pretrained checkpoints are available on [OneDrive](https://1drv.ms/f/c/e762fb0a44e578db/IgC3MzxqVlM6SpRyUbQOFn0UAVwPjLYXZN4x3jPqNflMt6s?e=28xlmR). Download them into the matching experiment folders before running inference.

Checkpoint paths follow the training log hierarchy:

```text
experiments/auto_encoder/
└── AE_LOG/
    ├── checkpoint/vxl_0_011_last.ckpt
    └── generator/
        └── GEN_LOG/
            ├── checkpoint/vxl_0_011_last.ckpt
            └── control/
                └── CONTROL_LOG/
                    └── checkpoint/vxl_0_011_last.ckpt
```

`AE_LOG` is the VAE run folder, `GEN_LOG` is the Flow Matching generator run under that VAE, and `CONTROL_LOG` is the ControlNet completion run under that generator.

- Generation requires the VAE checkpoint (`AE_LOG`) and Flow Matching generator checkpoint (`GEN_LOG`).
- Completion requires the VAE checkpoint (`AE_LOG`), Flow Matching generator checkpoint (`GEN_LOG`), and ControlNet checkpoint (`CONTROL_LOG`).

## Inference

Inference uses the same entry point as training. Select a non-training task subcommand and pass the experiment folder names used for the VAE, generator, and ControlNet checkpoints. Set `--slurm.cluster local` to run immediately on the current machine.

```bash
# Layout-conditioned patch generation with a trained Flow Matching generator.
# By default, generation uses bounding boxes as the layout condition.
python -m seen2scene.main generator task:generation \
  --ae-log AE_LOG \
  --ckpt-path GEN_LOG \
  --latent-key tsdf_p_1.0 \
  --task.num-samples 10 \
  --task.repeat-num 1 \
  --task.export-as bbox mesh \
  --slurm.cluster local

# Larger scenes use sliding-window generation over 256^3 patches.
python -m seen2scene.main generator task:large-scale-generation \
  --ae-log AE_LOG \
  --ckpt-path GEN_LOG \
  --latent-key tsdf_p_1.0 \
  --task.scene-shape 384 384 256 \
  --task.overlap 0.2 \
  --task.cpu-offload \
  --task.num-samples 10 \
  --slurm.cluster local

# Partial-scan completion with a trained ControlNet.
python -m seen2scene.main control task:completion \
  --ae-log AE_LOG \
  --gen-log GEN_LOG \
  --ckpt-path CONTROL_LOG \
  --src-key tsdf_p_0.1 \
  --latent-key tsdf_p_1.0 \
  --task.num-samples 10 \
  --task.repeat-num 1 \
  --task.export-as bbox mesh volume \
  --slurm.cluster local
```

`AE_LOG`, `GEN_LOG`, and `CONTROL_LOG` are folder names under `experiments/auto_encoder`, `experiments/auto_encoder/AE_LOG/generator`, and `experiments/auto_encoder/AE_LOG/generator/GEN_LOG/control`. Outputs are written under a new timestamped inference directory in the same experiment tree; each scene export includes meshes, bounding boxes, optional volumes, and an `image_grid.png`.

Useful inference options:

- `--data.scene-names SCENE_ID ...` runs specific scenes.
- `--data.csv-path PATH` and `--data.csv-slice START END` run deterministic patch lists.
- `--task.guidance-scale 3.0` controls classifier-free guidance.
- `--task.overwrite` regenerates existing scene outputs.
- `--task.backend blender` uses Blender rendering instead of the faster `pyrender` backend.

## BibTeX

```bibtex
@misc{meng2026seen2scene,
      title={Seen2Scene: Completing Realistic 3D Scenes with Visibility-Guided Flow},
      author={Quan Meng and Yujin Chen and Lei Li and Matthias Nießner and Angela Dai},
      year={2026},
      eprint={2603.28548},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.28548},
}
```

## Acknowledgements

This project is mainly built on and inspired by [SG-NN](https://github.com/angeladai/sgnn), [XCube](https://github.com/nv-tlabs/XCube), [fVDB](https://github.com/openvdb/fvdb-core), and [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2). We also build on [VDBFusion](https://github.com/PRBonn/vdbfusion) for TSDF fusion and [BlenderProc](https://github.com/DLR-RM/BlenderProc) for synthetic scan rendering. We thank the authors and maintainers of these projects for releasing and maintaining their code, and thank the creators of [3D-FRONT/3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset), [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/), and [ARKitScenes](https://github.com/apple/ARKitScenes) for making their datasets available to the research community.
