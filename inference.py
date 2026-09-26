import argparse, cv2, torch, numpy as np
from model import NightDehazeNet

p=argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--output', default='outputs/enhanced.png')
p.add_argument('--checkpoint', default='checkpoints/nightdehaze_prototype.pth')
a=p.parse_args()

device='cuda' if torch.cuda.is_available() else 'cpu'
model=NightDehazeNet().to(device)
model.load_state_dict(torch.load(a.checkpoint,map_location=device)['model'])
model.eval()
img=cv2.imread(a.input)
if img is None: raise FileNotFoundError(a.input)
rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
h,w=rgb.shape[:2]
z=cv2.resize(rgb,(64,64)).astype('float32')/255
x=torch.from_numpy(z.transpose(2,0,1))[None].to(device)
with torch.no_grad(): out=model(x)[0].cpu().numpy().transpose(1,2,0)
out=cv2.resize(np.clip(out*255,0,255).astype('uint8'),(w,h),interpolation=cv2.INTER_CUBIC)
import os; os.makedirs(os.path.dirname(a.output) or '.',exist_ok=True)
cv2.imwrite(a.output,cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
print(a.output)
