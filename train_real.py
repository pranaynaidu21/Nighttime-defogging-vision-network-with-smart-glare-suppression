import os, glob, argparse, cv2, torch
from torch.utils.data import Dataset, DataLoader
from model import NightDehazeNet

class PairedDataset(Dataset):
    def __init__(self, root, size=64):
        self.size=size
        hazy=sorted(glob.glob(os.path.join(root,'train','hazy','*')))
        clear={os.path.basename(x):x for x in glob.glob(os.path.join(root,'train','clear','*'))}
        self.pairs=[(h,clear[os.path.basename(h)]) for h in hazy if os.path.basename(h) in clear]
        if not self.pairs: raise RuntimeError('No matching train/hazy and train/clear pairs found')
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        h,c=self.pairs[i]
        x=cv2.cvtColor(cv2.imread(h),cv2.COLOR_BGR2RGB)
        y=cv2.cvtColor(cv2.imread(c),cv2.COLOR_BGR2RGB)
        x=cv2.resize(x,(self.size,self.size)).astype('float32')/255
        y=cv2.resize(y,(self.size,self.size)).astype('float32')/255
        return torch.from_numpy(x.transpose(2,0,1)),torch.from_numpy(y.transpose(2,0,1))

p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch',type=int,default=8); p.add_argument('--out',default='checkpoints/nightdehaze_real.pth'); a=p.parse_args()
device='cuda' if torch.cuda.is_available() else 'cpu'
model=NightDehazeNet().to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); loss_fn=torch.nn.L1Loss()
loader=DataLoader(PairedDataset(a.data),batch_size=a.batch,shuffle=True,num_workers=0)
for e in range(a.epochs):
    total=0; model.train()
    for x,y in loader:
        x,y=x.to(device),y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step(); total+=loss.item()
    print(f'epoch {e+1}/{a.epochs} loss={total/len(loader):.4f}')
os.makedirs(os.path.dirname(a.out),exist_ok=True); torch.save({'model':model.state_dict(),'base':24,'training':'real paired nighttime dataset'},a.out)
print('saved',a.out)
