import os, argparse, torch
from torch.utils.data import Dataset, DataLoader
from model import NightDehazeNet
from synthetic_data import make_clear, degrade

class SyntheticDataset(Dataset):
    def __init__(self, n=240): self.n=n
    def __len__(self): return self.n
    def __getitem__(self, i):
        clear = make_clear(seed=i)
        hazy = degrade(clear, seed=i+10000)
        return torch.from_numpy(hazy.transpose(2,0,1)), torch.from_numpy(clear.transpose(2,0,1))

p=argparse.ArgumentParser()
p.add_argument('--epochs', type=int, default=3)
p.add_argument('--batch', type=int, default=8)
p.add_argument('--out', default='checkpoints/nightdehaze_prototype.pth')
a=p.parse_args()

device='cuda' if torch.cuda.is_available() else 'cpu'
model=NightDehazeNet().to(device)
opt=torch.optim.Adam(model.parameters(), lr=2e-3)
loader=DataLoader(SyntheticDataset(), batch_size=a.batch, shuffle=True, num_workers=0)
loss_fn=torch.nn.L1Loss()

for epoch in range(a.epochs):
    total=0; model.train()
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step(); total += loss.item()
    print(f'epoch {epoch+1}/{a.epochs} loss={total/len(loader):.4f}')

os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save({'model':model.state_dict(),'base':24,'training':'synthetic prototype pairs'}, a.out)
print('saved', a.out, 'on', device)
