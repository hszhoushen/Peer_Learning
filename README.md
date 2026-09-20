# Peer Learning for Unbiased Scene Graph Generation

Official PyTorch implementation of:

> **Peer Learning Approach to Unbiased Scene Graph Generation for Traffic Scene Understanding**
> Liguang Zhou, Junjie Hu, Yuhongze Zhou, Tin Lun Lam, and Yangsheng Xu
> *IEEE Transactions on Intelligent Transportation Systems*, 27(2):2365-2379, 2026
> [DOI: 10.1109/TITS.2025.3635279](https://doi.org/10.1109/TITS.2025.3635279)

## Overview

Peer Learning addresses the long-tailed predicate distribution in scene graph generation (SGG) through three specialized peers. Training samples are divided into head, body, and tail-oriented subsets, and the peers exchange knowledge through a peer-learning objective. At inference time, their predictions are combined by expertise-aware consensus voting.

The implementation supports:

- three SGG backbones: Motifs, VCTree, and Transformer;
- three Visual Genome protocols: PredCls, SGCls, and SGDet;
- Open Images V6 SGDet experiments;
- configurable peer number, sample-partition mode, random seed, and voting weights.

This repository is based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). The original baseline and experimental scripts are retained for reference, while the reproducible Peer Learning entry points are under `scripts/PLME/`.

## Installation

The original experiments use Python 3.7, PyTorch 1.4.0, torchvision 0.5.0, CUDA 10.1, and NVIDIA Apex mixed precision. See [INSTALL.md](INSTALL.md) for the complete legacy environment setup.

```bash
git clone https://github.com/hszhoushen/Peer_Learning.git
cd Peer_Learning
python setup.py build develop
```

## Data Preparation

Follow [DATASET.md](DATASET.md) to prepare Visual Genome. The default configuration expects:

- Visual Genome images and annotations;
- GloVe embeddings under `./glove`;
- a pretrained Faster R-CNN checkpoint at `./checkpoints/pretrained_faster_rcnn/model_final.pth`.

Alternative locations can be supplied through the environment variables described below. Dataset files and model checkpoints are not stored in Git.

## Quick Start

All standard experiments use the common runner `scripts/PLME/run.sh`. The wrappers select the paper configuration for each backbone and task.

Train Motifs on PredCls with two GPUs:

```bash
GPUS=0,1 SEED=678 bash scripts/PLME/motifs/train_predcls.sh
```

Evaluate a trained checkpoint directory:

```bash
GPUS=0,1 \
OUTPUT_DIR=./checkpoints/motifs-pl-predcls-seed42 \
bash scripts/PLME/motifs/test_predcls.sh
```

The output directory created during training contains a `last_checkpoint` file and is loaded automatically. To evaluate a standalone checkpoint, append an explicit override such as `MODEL.WEIGHT /path/to/model_final.pth`.

Run the corresponding VCTree or Transformer experiment:

```bash
GPUS=0,1 SEED=678 bash scripts/PLME/vctree/train_predcls.sh
GPUS=0,1 SEED=678 bash scripts/PLME/transformer/train_predcls.sh
```

The same interface is available for `predcls`, `sgcls`, and `sgdet`, for both training and testing:

```text
scripts/PLME/<motifs|vctree|transformer>/<train|test>_<predcls|sgcls|sgdet>.sh
```

### Runtime Options

The wrappers can be configured without editing source files:

| Variable | Default | Description |
| --- | --- | --- |
| `GPUS` | `0,1` | Comma-separated visible GPU IDs |
| `SEED` | `678` | Default paper seed used for training; override it for multi-run studies |
| `GLOVE_DIR` | `./glove` | GloVe embedding directory |
| `DETECTOR_CKPT` | `./checkpoints/pretrained_faster_rcnn/model_final.pth` | Detector checkpoint |
| `OUTPUT_DIR` | task-dependent | Training output or evaluation checkpoint directory |
| `NUM_EXPERTS` | `3` | Number of peers |
| `EXPERT_MODE` | `hbt_b_t` | Head/body/tail sample-partition mode |
| `KNOWLEDGE_WEIGHTS` | backbone-dependent | Expertise-aware voting weights |
| `BASE_LR` | `0.01` (`0.001` for Transformer) | Base learning rate |
| `TRAIN_BATCH` | `12` | Global training batch size |
| `MAX_ITER` | `20000` | Maximum training iterations |
| `SYNC_GATHER` | `False` | Use index-safe object gathering during distributed evaluation |
| `DRY_RUN` | `0` | Print the resolved command without launching it |

Additional configuration overrides can be appended to any wrapper command, for example:

```bash
GPUS=2,3 SEED=379 MAX_ITER=24000 \
bash scripts/PLME/motifs/train_predcls.sh SOLVER.VAL_PERIOD 1000
```

When changing `NUM_EXPERTS`, also provide an `EXPERT_MODE` and a `KNOWLEDGE_WEIGHTS` list of matching length.

`MODEL.ROI_RELATION_HEAD.KNOWLEDGE_WEIGHTS` is now read from YAML or command-line configuration and is no longer overwritten inside the training program.

## Open Images V6

Open Images experiments use the Transformer backbone and the SGDet protocol:

```bash
GPUS=0,1 SEED=678 bash scripts/PLME/openimages/train_sgdet.sh
GPUS=0,1 OUTPUT_DIR=./checkpoints/transformer-pl-oiv6-sgdet-seed678 \
bash scripts/PLME/openimages/test_sgdet.sh
```

Set `CONFIG_FILE`, `GLOVE_DIR`, `DETECTOR_CKPT`, and `OUTPUT_DIR` if the files are stored elsewhere.

## Selected Results

The paper reports substantial improvements in mean Recall under the standard Visual Genome protocols. Selected mR@100 results are shown below; consult the paper for the complete R@K, mR@K, and backbone comparisons.

| Backbone | PredCls mR@100 | SGCls mR@100 | SGDet mR@100 |
| --- | ---: | ---: | ---: |
| Motifs + Peer Learning | 40.9 | 20.9 | 19.2 |
| VCTree + Peer Learning | 42.1 | 26.5 | 19.8 |

Release validation was also performed on all 26,446 Visual Genome PredCls test images using the published `model_0020000.pth` checkpoints:

| Backbone | R@20/50/100 | mR@20/50/100 |
| --- | ---: | ---: |
| Motifs + Peer Learning | 34.68 / 38.90 / 39.91 | 31.84 / 38.75 / 40.94 |
| VCTree + Peer Learning | 39.25 / 44.85 / 46.42 | 32.70 / 39.35 / 41.93 |

The Motifs result reproduces the original log within 0.02 percentage points. The VCTree checkpoint preserves the expected strong mean Recall, with a small shift in the mR--R trade-off under the current software environment and explicit expertise-aware voting weights. Small differences can result from the CUDA/PyTorch version, distributed evaluation, voting configuration, and random seed. For statistical studies, keep all settings fixed and change only `SEED`.

## Checkpoints

Large detector and SGG checkpoints are intentionally excluded from the Git repository. Published model files can be attached to a GitHub release without changing the source history. Each released model should include its backbone, task, seed, configuration, and reported metrics. Use `v1.0.1-tits` or later for the complete dataset loaders, current NumPy compatibility fixes, and index-safe multi-GPU evaluation defaults.

## Citation

If this project is useful in your research, please cite:

```bibtex
@article{zhou2026peer,
  author  = {Zhou, Liguang and Hu, Junjie and Zhou, Yuhongze and Lam, Tin Lun and Xu, Yangsheng},
  title   = {Peer Learning Approach to Unbiased Scene Graph Generation for Traffic Scene Understanding},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2026},
  volume  = {27},
  number  = {2},
  pages   = {2365--2379},
  doi     = {10.1109/TITS.2025.3635279}
}
```

## Acknowledgments

We thank the authors of [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), Neural Motifs, and VCTree for their open-source implementations.

## License

This project follows the license included in [LICENSE](LICENSE).
