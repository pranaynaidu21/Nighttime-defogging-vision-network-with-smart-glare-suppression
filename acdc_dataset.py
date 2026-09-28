import os
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg"}


class ACDCPairedDataset(Dataset):
    """Paired ACDC adverse-condition -> normal-condition dataset.

    V2 uses paired random patches instead of shrinking an entire road image to
    128x128. This preserves road markings, vehicles, trees and light details.

    Expected root:
        E:/rgb_anon_trainvaltest/rgb_anon
    with fog/night/rain/snow/{train,train_ref,val,val_ref,test,test_ref}.
    """

    def __init__(self, root, split="train", conditions=("fog", "night", "rain", "snow"),
                 size=256, augment=False, synthetic_glare=False):
        self.root = Path(root)
        self.split = split
        self.conditions = tuple(conditions)
        self.size = int(size)
        self.augment = augment
        self.synthetic_glare = synthetic_glare
        self.pairs = []

        for condition in self.conditions:
            input_dir = self.root / condition / split
            ref_dir = self.root / condition / f"{split}_ref"
            if not input_dir.exists():
                raise FileNotFoundError(f"Missing ACDC input directory: {input_dir}")
            if not ref_dir.exists():
                raise FileNotFoundError(f"Missing ACDC reference directory: {ref_dir}")

            refs = {}
            for p in ref_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    refs[p.name.replace("_rgb_ref_anon", "_rgb_anon")] = p

            for p in input_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS and p.name in refs:
                    self.pairs.append((p, refs[p.name], condition))

        if not self.pairs:
            raise RuntimeError(
                f"No ACDC pairs found under {self.root} for split={split}, conditions={conditions}"
            )

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _read(path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _crop_pair(self, x, y):
        h, w = x.shape[:2]
        s = self.size
        if h < s or w < s:
            scale = max(s / h, s / w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            x = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA)
            y = cv2.resize(y, (nw, nh), interpolation=cv2.INTER_AREA)
            h, w = x.shape[:2]

        if self.split == "train":
            top = random.randint(0, h - s)
            left = random.randint(0, w - s)
        else:
            top = max(0, (h - s) // 2)
            left = max(0, (w - s) // 2)

        return x[top:top + s, left:left + s], y[top:top + s, left:left + s]

    def _degrade(self, x):
        if random.random() < 0.55:
            gamma = random.uniform(0.65, 1.15)
            x = np.power(np.clip(x, 0, 1), gamma)
        if random.random() < 0.35:
            x *= random.uniform(0.70, 0.95)

        # Synthetic headlight / street-light glare for the prototype.
        if self.synthetic_glare and random.random() < 0.35:
            h, w = x.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w]
            for _ in range(random.randint(1, 2)):
                cx = random.randint(0, max(0, w - 1))
                cy = random.randint(0, max(0, h - 1))
                sigma = random.uniform(8, max(10, min(h, w) * 0.08))
                strength = random.uniform(0.10, 0.35)
                glow = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
                x = np.clip(x + glow[..., None] * strength, 0, 1)

        if random.random() < 0.25:
            noise = np.random.normal(0, random.uniform(0.003, 0.015), x.shape).astype(np.float32)
            x = np.clip(x + noise, 0, 1)
        return x

    def __getitem__(self, index):
        inp_path, ref_path, condition = self.pairs[index]
        x = self._read(inp_path).astype(np.float32) / 255.0
        y = self._read(ref_path).astype(np.float32) / 255.0

        x, y = self._crop_pair(x, y)

        if self.augment:
            if random.random() < 0.5:
                x = np.ascontiguousarray(x[:, ::-1])
                y = np.ascontiguousarray(y[:, ::-1])
            x = self._degrade(x)

        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1)))
        return x, y, condition, str(inp_path)
