import argparse
import os
import cv2
import numpy as np
import torch
from model import NightDehazeNet


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    model = NightDehazeNet(base=ckpt.get('base', 24)).to(device).float()
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, int(ckpt.get('size', 256))


def enhance_tiled(model, rgb, device, tile=256, overlap=64):
    """Restore a full-resolution image with overlapping tiles."""
    h, w = rgb.shape[:2]
    if tile % 8 != 0:
        raise ValueError('tile must be divisible by 8')
    step = max(1, tile - overlap)
    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)
    padded = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    ph, pw = padded.shape[:2]
    acc = np.zeros((ph, pw, 3), dtype=np.float32)
    weight = np.zeros((ph, pw, 1), dtype=np.float32)

    yy, xx = np.mgrid[0:tile, 0:tile]
    cy = (tile - 1) / 2.0
    cx = (tile - 1) / 2.0
    sigma = max(1.0, tile * 0.30)
    win = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)).astype(np.float32)
    win = (0.25 + 0.75 * win)[..., None]

    with torch.no_grad():
        for top in range(0, max(1, ph - tile + 1), step):
            if top + tile > ph:
                top = ph - tile
            for left in range(0, max(1, pw - tile + 1), step):
                if left + tile > pw:
                    left = pw - tile
                patch = padded[top:top + tile, left:left + tile].astype(np.float32) / 255.0
                tensor = torch.from_numpy(patch.transpose(2, 0, 1))[None].to(device=device, dtype=torch.float32)
                out = model(tensor)[0].cpu().numpy().transpose(1, 2, 0)
                acc[top:top + tile, left:left + tile] += out * win
                weight[top:top + tile, left:left + tile] += win

    return np.clip(acc / np.maximum(weight, 1e-6), 0, 1)[:h, :w]


def laplacian_sharpness(rgb01):
    gray = cv2.cvtColor(np.clip(rgb01 * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def safe_quality_guard(original, restored):
    """Prevent extreme degradation from being presented as an enhancement."""
    orig_sharp = laplacian_sharpness(original)
    out_sharp = laplacian_sharpness(restored)
    change = float(np.mean(np.abs(restored - original)))
    if out_sharp < orig_sharp * 0.55 or change > 0.20:
        restored = 0.65 * original + 0.35 * restored
        guarded = True
    else:
        guarded = False
    return np.clip(restored, 0, 1), orig_sharp, out_sharp, change, guarded


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', default='outputs/enhanced.png')
    p.add_argument('--checkpoint', default='checkpoints/acdc_v2_best.pth')
    p.add_argument('--tile', type=int, default=256)
    p.add_argument('--overlap', type=int, default=64)
    a = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, trained_size = load_model(a.checkpoint, device)
    tile = a.tile or trained_size
    img = cv2.imread(a.input)
    if img is None:
        raise FileNotFoundError(a.input)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    restored = enhance_tiled(model, rgb, device, tile=tile, overlap=a.overlap)
    restored, orig_sharp, out_sharp, change, guarded = safe_quality_guard(
        rgb.astype(np.float32) / 255.0, restored
    )
    os.makedirs(os.path.dirname(a.output) or '.', exist_ok=True)
    cv2.imwrite(a.output, cv2.cvtColor((restored * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f'output={a.output}')
    print(f'original_sharpness={orig_sharp:.2f}')
    print(f'model_sharpness={out_sharp:.2f}')
    print(f'mean_change={change:.4f}')
    print(f'quality_guard={guarded}')


if __name__ == '__main__':
    main()
