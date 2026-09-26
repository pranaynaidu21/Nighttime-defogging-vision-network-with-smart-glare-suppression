# Dataset setup

For the academic experiment, use a paired nighttime dehazing dataset such as the NT-HAZE 2026 benchmark.

Official repository:
https://github.com/NTIRE-HAZE/Nighttime-Image-Dehazing-NTIRE-2026

Keep third-party dataset files outside GitHub. Expected layout for the provided real-data trainer:

```text
data/NT-HAZE/
  train/
    hazy/
    clear/
  val/
    hazy/
    clear/
```

Train:

```bash
python train_real.py --data data/NT-HAZE --epochs 10
```

The default `train.py` creates synthetic paired nighttime degradation data. This makes the prototype easy to demonstrate on a laptop without downloading a large dataset. Any synthetic metrics should be reported separately from real-dataset results.
