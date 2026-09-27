import argparse
import os
import cv2
import numpy as np
import torch
from model import NightDehazeNet

p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--output', default='outputs/enhanced.png')
p.add_argument('--checkpoint', default='checkpoints/acdc_best.pth')
p.add_argument('--size', type=int, default=128)
a = p.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(a.checkpoint, map_location=device)
model = NightDehazeNet(base=ckpt.get('base', 24)).to(device)
model.load_state_dict(ckpt['model'])
model.eval()

img = cv2.imread(a.input)
if img is None:
    raise FileNotFoundError(a.input)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = rgb.shape[:2]
z = cv2.resize(rgb, (a.size, a.size), interpolation=cv2.INTER_AREA).astype('float32') / 255.0
x = torch.from_numpy(z.transpose(2, 0, 1))[None].to(device)
with torch.no_grad():
    out = model(x)[0].cpu().numpy().transpose(1, 2, 0)
out = cv2.resize(np.clip(out * 255, 0, 255).astype('uint8'), (w, h), interpolation=cv2.INTER_CUBIC)
os.makedirs(os.path.dirname(a.output) or '.', exist_ok=True)
cv2.imwrite(a.output, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print(a.output)
