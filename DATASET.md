# Dataset setup

## Recommended real nighttime dataset

**NT-HAZE 2026** is an official nighttime image dehazing benchmark with paired real hazy and haze-free images captured across 40 nighttime indoor scenes. Official source:

https://github.com/NTIRE-HAZE/Nighttime-Image-Dehazing-NTIRE-2026

Expected layout for the provided trainer:

```text
data/NT-HAZE/
  train/
    hazy/
    clear/
  val/
    hazy/
    clear/
```

Train with:

```bash
python train_real.py --data data/NT-HAZE --epochs 10
```

## Alternative published nighttime data

The ACMMM 2023 `jinyeying/nighttime_dehaze` project documents **RealNightHaze (443 hazy images)**, clean nighttime reference collections, and GTA5 nighttime fog data, with download links in that project's README:

https://github.com/jinyeying/nighttime_dehaze

Use the dataset license/terms and the original download instructions. Do not commit third-party datasets to this repository.

## Prototype training

The default `train.py` creates small synthetic nighttime degradation pairs locally. This makes the demo easy to run on a CPU laptop. Synthetic metrics must be reported separately from real-dataset benchmark results.
