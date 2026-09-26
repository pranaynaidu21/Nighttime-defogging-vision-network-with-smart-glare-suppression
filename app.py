import streamlit as st
import cv2, torch, numpy as np
from PIL import Image
from model import NightDehazeNet

st.set_page_config(page_title='NightVision-Dehaze', layout='wide')
st.title('NightVision-Dehaze')
st.caption('CNN + Attention + Transformer global context prototype')

@st.cache_resource
def load_model():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=NightDehazeNet().to(device)
    model.load_state_dict(torch.load('checkpoints/nightdehaze_prototype.pth',map_location=device)['model'])
    model.eval()
    return model,device

file=st.file_uploader('Upload a nighttime / foggy image', type=['jpg','jpeg','png'])
if file:
    img=np.array(Image.open(file).convert('RGB'))
    h,w=img.shape[:2]
    x=cv2.resize(img,(64,64)).astype('float32')/255
    model,device=load_model()
    with torch.no_grad():
        out=model(torch.from_numpy(x.transpose(2,0,1))[None].to(device))[0].cpu().numpy().transpose(1,2,0)
    out=np.clip(cv2.resize(out,(w,h),interpolation=cv2.INTER_CUBIC),0,1)
    left,right=st.columns(2)
    left.image(img,caption='Input',use_container_width=True)
    right.image(out,caption='Enhanced',use_container_width=True)
    png=cv2.imencode('.png',(out*255).astype('uint8'))[1].tobytes()
    st.download_button('Download enhanced image',png,'enhanced.png','image/png')
