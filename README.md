# NightVision-Dehaze

**Nighttime Image De-Fogging with Smart Glare Suppression using CNN and Transformer Features**

A laptop-friendly research prototype for nighttime/foggy image enhancement. The pipeline combines CNN local features, channel attention, Transformer-style global context, feature fusion, and PSNR/SSIM evaluation.

## Prototype pipeline

Input nighttime/foggy image → CNN local features → attention → Transformer global context → feature fusion → enhanced image.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python train.py --epochs 3
streamlit run app.py
```

Command-line inference:

```bash
python inference.py --input data/sample_night_foggy.png --output outputs/enhanced.png
```

## Dataset

The repository deliberately does not commit third-party datasets. `DATASET.md` describes how to use a paired nighttime dehazing dataset and how to train with `train_real.py`.

The default training script generates small synthetic nighttime degradation pairs so the prototype can run on a CPU laptop without a large download.

## Research note

This is a proof-of-concept implementation inspired by the project methodology. It is not a claim that the prototype reproduces a published architecture exactly. Report real-dataset PSNR/SSIM separately from synthetic prototype results.
