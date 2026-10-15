# STGSM

Implementation of **STGSM** for micro-level information diffusion prediction (next-user prediction). It couples a
multi-granularity hypergraph encoder with a diffusion-inspired conditional target generator and a
spatio-temporal guiding cross-attention module.

This repository contains the model, the training/evaluation entry point, and one trained
checkpoint per dataset.

---

## Environment

```bash
pip install -r requirements.txt
```

The pinned dependencies target Python 3.12, PyTorch 2.8.0 (CUDA 12.8) and torch-geometric 2.8.0.

## Data

Point the loader at your data root; every dataset path is derived from it:

```bash
export STGSM_DATA_ROOT=/path/to/data/
```

The layout below `STGSM_DATA_ROOT` is one directory per dataset, named `android`,
`christianity`, `twitter`, `memetracker`:

```
$STGSM_DATA_ROOT/<dataset>/
├── cascade.txt              # training cascades
├── cascadevalid.txt         # validation cascades
├── cascadetest.txt          # test cascades
├── cascade_id.txt           # training cascade IDs, in matching line order
├── cascadevalid_id.txt      # validation cascade IDs
├── cascadetest_id.txt       # test cascade IDs
├── u2idx.pickle             # user-to-index mapping
├── idx2u.pickle             # inverse user mapping
├── ui2idx.pickle            # combined user/cascade mapping
├── idx2ui.pickle            # inverse combined mapping
├── edges.txt                # input social graph, when available
├── edges_refined.txt        # processed social graph, when available
├── edges_item.txt           # training user/cascade graph
└── edges_reposts.txt        # training diffusion graph
```

Core hyperparameters can be changed directly, for example:

```bash
python train.py --data twitter --device cuda --epoch 50 --seed 2023 \
    --lr 0.001 --d_model 128 --lightgcn_layers 3 \
    --save_path checkpoints/twitter_retrained.pt
```

Run `python train.py --help` for the complete parameter list.

```

### The four checkpoints included in `checkpoints/`

```bash
python train.py --data christianity --device cuda --eval_only --save_path checkpoints/christianity.pt
python train.py --data android --device cuda --eval_only --save_path checkpoints/android.pt
python train.py --data twitter --device cuda --eval_only --save_path checkpoints/twitter.pt
python train.py --data memetracker --device cuda --eval_only --save_path checkpoints/memetracker.pt
```
