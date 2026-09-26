import argparse, cv2, torch, numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from model import NightDehazeNet

p=argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--reference',required=True)
p.add_argument('--checkpoint',default='checkpoints/nightdehaze_prototype.pth')
a=p.parse_args()

device='cuda' if torch.cuda.is_available() else 'cpu'
model=NightDehazeNet().to(device)
model.load_state_dict(torch.load(a.checkpoint,map_location=device)['model']); model.eval()
x=cv2.cvtColor(cv2.imread(a.input),cv2.COLOR_BGR2RGB)
r=cv2.cvtColor(cv2.imread(a.reference),cv2.COLOR_BGR2RGB)
h,w=r.shape[:2]
z=cv2.resize(x,(64,64)).astype('float32')/255
with torch.no_grad(): y=model(torch.from_numpy(z.transpose(2,0,1))[None].to(device))[0].cpu().numpy().transpose(1,2,0)
y=cv2.resize(np.clip(y,0,1),(w,h),interpolation=cv2.INTER_CUBIC)
print(f'PSNR: {peak_signal_noise_ratio(r/255.0,y,data_range=1.0):.4f} dB')
print(f'SSIM: {structural_similarity(r/255.0,y,channel_axis=2,data_range=1.0):.4f}')
