# Training

## Laptop prototype

Run:

```bash
python train.py --epochs 3
```

The script generates paired synthetic nighttime examples locally. This avoids requiring a large dataset for the first demonstration.

## Real nighttime data

A published nighttime dehazing repository reports RealNightHaze with 443 hazy images and additional clean-reference collections, as well as GTA5 nighttime fog data. See `DATASET.md` for the source. The repository also provides evaluation scripts/results for several nighttime datasets. Use the source's license/terms before downloading or redistributing any data.

For the NT-HAZE 2026 benchmark, use its official repository and follow its download instructions. It provides paired real hazy and haze-free nighttime images organized into train/validation splits.

## Recommended experiment

1. Download the dataset yourself.
2. Keep it outside GitHub.
3. Put matching `hazy` and `clear` files under the expected directory structure.
4. Run `python train_real.py --data <dataset-root> --epochs 10`.
5. Evaluate on held-out references with `evaluate.py`.
6. Report real-data PSNR/SSIM separately from synthetic prototype results.
