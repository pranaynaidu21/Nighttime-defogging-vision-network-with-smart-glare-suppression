# NightVision-Dehaze

**Nighttime Image De-Fogging with Smart Glare Suppression using CNN and Transformer Features**

A laptop-friendly research prototype for nighttime/foggy road-image enhancement. The model combines CNN local features, channel attention, Transformer global context, skip-connection feature fusion, and PSNR/SSIM evaluation.

## ACDC training

The project supports the paired ACDC adverse-condition dataset. Keep the dataset outside this repository.

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
└── night
    ├── train
    ├── train_ref
    ├── val
    ├── val_ref
    ├── test
    └── test_ref
```

The loader matches `*_rgb_anon.png` inputs with the corresponding `*_rgb_ref_anon.png` normal-condition images. Training uses fog + night pairs. Synthetic brightness reduction, noise, and headlight/street-light glare are applied only as training augmentation.

## Install

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the ACDC prototype

Quick smoke test with a small subset:

```powershell
python train_acdc.py --data "E:\rgb_anon_trainvaltest\rgb_anon" --epochs 1 --limit 20 --batch 2 --size 128
```

Full laptop prototype training:

```powershell
python train_acdc.py --data "E:\rgb_anon_trainvaltest\rgb_anon" --epochs 5 --batch 2 --size 128
```

The best checkpoint is written to:

```text
checkpoints/acdc_best.pth
```

The script reports training/validation loss, PSNR, and SSIM after each epoch.

## Run the Streamlit demo

After training:

```powershell
streamlit run app.py
```

Upload a foggy or nighttime road image to compare the input and enhanced output.

## Command-line inference

```powershell
python inference.py --input "path\to\image.png" --output "outputs\enhanced.png"
```

## Dataset policy

Third-party datasets are intentionally not committed to GitHub. Keep ACDC on the local machine and use its own license/terms. Only source code, documentation, and small project assets belong in this repository.

## Research note

This is a proof-of-concept implementation, not a claim that it reproduces a published architecture exactly. Report real ACDC test metrics separately from synthetic augmentation results.
