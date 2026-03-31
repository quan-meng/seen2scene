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
