# NightVision-Dehaze v2

**Nighttime and adverse-weather road-image restoration with CNN, channel attention and Transformer features**

A laptop-friendly research prototype for enhancing nighttime, foggy, rainy and snowy road images. V2 keeps the input resolution intact during inference by processing overlapping 256x256 tiles instead of shrinking the entire image to 128x128.

## What changed in v2

- ACDC **fog + night + rain + snow** paired training.
- Random **256x256 patches** during training to preserve local road/object details.
- Conservative residual restoration so the model learns a correction instead of freely replacing the input.
- L1 + SSIM + Sobel edge loss to reduce blur and preserve edges.
- Full-resolution tiled inference with overlap blending.
- Quality guard that blends toward the original if the neural output becomes substantially softer or changes too aggressively.
- Lightweight LAB contrast/sharpening as an optional final post-processing stage in the Streamlit app.

## ACDC dataset

Keep the dataset outside this repository.

Expected local dataset root:

```text
E:\rgb_anon_trainvaltest\rgb_anon
├── fog
│   ├── train
│   ├── train_ref
│   ├── val
│   ├── val_ref
│   ├── test
│   └── test_ref
├── night
│   ├── train
│   ├── train_ref
│   ├── val
│   ├── val_ref
│   ├── test
│   └── test_ref
├── rain
│   ├── train
│   ├── train_ref
│   ├── val
│   ├── val_ref
│   ├── test
│   └── test_ref
└── snow
    ├── train
    ├── train_ref
    ├── val
    ├── val_ref
    ├── test
    └── test_ref
```

The loader matches `*_rgb_anon.png` inputs with the corresponding `*_rgb_ref_anon.png` normal-condition images.

## Install

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train v2 locally

Start with 100 pairs to verify the pipeline:

```powershell
python train_acdc.py --data "E:\rgb_anon_trainvaltest\rgb_anon" --epochs 2 --limit 100 --batch 2 --size 256
```

Then train the 500-pair prototype:

```powershell
python train_acdc.py --data "E:\rgb_anon_trainvaltest\rgb_anon" --epochs 5 --limit 500 --batch 2 --size 256
```

For the final prototype, you can remove `--limit 500` to use all available ACDC training pairs:

```powershell
python train_acdc.py --data "E:\rgb_anon_trainvaltest\rgb_anon" --epochs 5 --batch 2 --size 256
```

V2 saves the best checkpoint to:

```text
checkpoints/acdc_v2_best.pth
```

The script reports training/validation loss, PSNR and SSIM after every epoch.

## Run the Streamlit demo

After v2 training:

```powershell
streamlit run app.py
```

Upload a nighttime, foggy, rainy or snowy road image. The app shows:

1. Original input.
2. Full-resolution neural restoration with the quality guard.
3. Final lightweight enhanced output.

It also displays sharpness and mean-change diagnostics.

## Command-line inference

```powershell
python inference.py --input "path\to\image.png" --output "outputs\enhanced_v2.png"
```

The command uses 256x256 overlapping tiles by default and writes quality diagnostics to the terminal.

## Dataset policy

Third-party datasets are intentionally not committed to GitHub. Keep ACDC on the local machine and follow its license/terms. Only source code, documentation, and small project assets belong in this repository.

## Research note

This is a proof-of-concept implementation, not a claim that it reproduces a published architecture exactly. Report real ACDC test metrics separately from synthetic augmentation results.
