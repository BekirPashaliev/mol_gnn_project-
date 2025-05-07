# mol\_gnn\_project – Graph‑based Drug Discovery Pipeline

> **Purpose** – end‑to‑end research code for learning a latent representation of small‑molecule graphs, predicting their IC₅₀ against a chosen cancer target, and generating novel candidate structures with improved activity.

---

## 1 · Motivation

Traditional virtual screening relies on handcrafted fingerprints and brute‑force docking.  We replace that with a *graph variational auto‑encoder* (VAE) that learns chemistry directly from molecular graphs, a supervised regressor that maps the latent space to bio‑activity, and an optimizer that navigates that space to propose new molecules.

---

## 2 · High‑level Roadmap

| Phase | Goal                                               | Key artefacts                                 |
| ----- |----------------------------------------------------| --------------------------------------------- |
| **A** | Train Graph VAE on raw structures                  | `checkpoints/vae_best.ckpt`, TensorBoard logs |
| **B** | Freeze encoder, fit regressor on `(z, IC₅₀)` pairs | `checkpoints/reg_best.ckpt`                   |
| **C** | Search latent space  ⇒ generate candidate SMILES   | `outputs/candidates.smi`                      |
| **D** | *Outside scope*: dock, MD, synthesize, assay       | —                                             |

---

## 3 · Repository Layout

```text
mol_gnn_project/
├── configs/               # YAML hyper‑parameters
├── data/                  # loading, cleaning, splitting, GraphDataset
├── graphs/                # RDKit → PyG graph conversion, featurizers
├── models/                # VAE, IC50 regressor, losses
├── training/              # train loops for VAE / regressor
├── tuning/                # Optuna search spaces & objectives
├── generation/            # latent‑space optimizers (grad, beam)
├── utils/                 # metrics, logging, checkpoint helpers
├── scripts/               # CLI entry‑points (preprocess, train, generate)
├── notebooks/             # exploratory analyses
└── tests/                 # unit tests
```

---

## 4 · Installation

```bash
# 1 – create virtual env (conda or venv)
conda create -n molgnn python=3.12
conda activate molgnn
# 2 – install deps
pip install -r requirements.txt  # RDKit‑pypi, torch, torch‑geometric…
```

GPU with CUDA 11+ is strongly recommended for model training.

---

## 5 · Quick Start

```bash
# 1. preprocess SMILES‑IC50 csv
python scripts/preprocess_data.py --cfg configs/default.yaml

# 2. train VAE (Phase A)
python scripts/train_vae.py --cfg configs/default.yaml

# 3. train IC50 regressor (Phase B)
python scripts/train_regressor.py --cfg configs/default.yaml \
    --encoder_ckpt checkpoints/vae_best.ckpt

# 4. generate new molecules (Phase C)
python scripts/generate_candidates.py --cfg configs/default.yaml \
    --vae_ckpt checkpoints/vae_best.ckpt \
    --reg_ckpt checkpoints/reg_best.ckpt
```

---

## 6 · Workflows in Detail

### 6.1 Data Pre‑processing

1. *Load raw CSV* – columns: `id, smiles, ic50`
2. *Clean SMILES* – strip salts, remove stereo if ambiguous
3. *Split* – stratified train/val/test (70/15/15)
4. *Graph build* – RDKit → PyG `Data` objects with atom/bond features.

### 6.2 Model Training

- **VAE** – reconstruction + β‑KLD losses, early stopping on val loss.
- **Regressor** – MSE on log₁₀(IC₅₀); encoder weights frozen.
- **Logging** – TensorBoard + MLflow; checkpoints saved on val metric improvement.

### 6.3 Candidate Generation

1. Seed latent codes with actives (or random).
2. *Gradient search* to minimise predicted IC₅₀.
3. Decode, deduplicate, validity check (RDKit).
4. Output ranked `candidates.smi` for downstream physics‑based screening.

---

## 7 · Contributing Guide

1. Create branch `feature/<name>`
2. Add or update **tests** in `tests/`
3. Ensure `pytest` & `ruff` pass.
4. Submit PR; CI will run lint + unit tests + sample training on 100 molecules.

---

## 8 · Planned Extensions

- Graph Transformer backbone.
- Semi‑supervised joint training (VAE + IC₅₀) to tighten latent/activity link.
- Integration with docking API for RL‑style feedback.

---

## 9 · License

Released under the MIT License – see `LICENSE` for details.

