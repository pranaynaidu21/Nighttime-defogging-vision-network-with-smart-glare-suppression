import os
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from model import NightDehazeNet

st.set_page_config(page_title='NightVision-Dehaze', layout='wide')
st.title('NightVision-Dehaze')
st.caption('ACDC-trained CNN + Channel Attention + Transformer prototype')

CHECKPOINT = 'checkpoints/acdc_best.pth'

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(CHECKPOINT):
        return None, device
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model = NightDehazeNet(base=ckpt.get('base', 24)).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, device

file = st.file_uploader('Upload a nighttime / foggy road image', type=['jpg', 'jpeg', 'png'])
model, device = load_model()

if model is None:
    st.warning('Train the ACDC model first: python train_acdc.py')
elif file:
    img = np.array(Image.open(file).convert('RGB'))
    h, w = img.shape[:2]
    size = 128
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype('float32') / 255.0
    tensor = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
    with torch.no_grad():
        out = model(tensor)[0].cpu().numpy().transpose(1, 2, 0)
    out = np.clip(cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)
    left, right = st.columns(2)
    left.image(img, caption='Input', use_container_width=True)
    right.image(out, caption='Enhanced', use_container_width=True)
    png = cv2.imencode('.png', (out * 255).astype('uint8'))[1].tobytes()
    st.download_button('Download enhanced image', png, 'enhanced.png', 'image/png')
