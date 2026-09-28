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


def enhance_postprocess(rgb, strength=0.65):
    """Lightweight laptop-friendly enhancement after the neural network.

    Uses LAB local contrast enhancement plus mild detail sharpening. The
    original/model result is blended with the enhancement so it is less
    likely to introduce aggressive artifacts.
    """
    img8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    # Local contrast enhancement on luminance only.
    lab = cv2.cvtColor(img8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    contrast = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2RGB)

    # Mild unsharp mask for road/vehicle/edge details.
    blurred = cv2.GaussianBlur(contrast, (0, 0), 1.2)
    sharpened = cv2.addWeighted(contrast, 1.20, blurred, -0.20, 0)

    # Blend rather than replacing the neural output completely.
    alpha = float(np.clip(strength, 0.0, 1.0))
    enhanced = cv2.addWeighted(img8, 1.0 - alpha, sharpened, alpha, 0)
    return np.clip(enhanced.astype(np.float32) / 255.0, 0, 1)


file = st.file_uploader(
    'Upload a nighttime / foggy road image',
    type=['jpg', 'jpeg', 'png']
)
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
        model_out = model(tensor)[0].cpu().numpy().transpose(1, 2, 0)

    # Restore the original image resolution.
    model_out = np.clip(
        cv2.resize(model_out, (w, h), interpolation=cv2.INTER_CUBIC),
        0,
        1,
    )

    st.subheader('Post-processing image enhancer')
    strength = st.slider(
        'Enhancement strength',
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
        help='Higher values add more local contrast and detail sharpening.'
    )
    enhanced = enhance_postprocess(model_out, strength)

    # Three-stage visual check: original -> neural model -> enhancer.
    col1, col2, col3 = st.columns(3)
    col1.image(img, caption='1. Original Input', use_container_width=True)
    col2.image(model_out, caption='2. Neural Model Output', use_container_width=True)
    col3.image(enhanced, caption='3. Final Enhanced Output', use_container_width=True)

    st.info(
        'The final image is a post-processed version of the neural model output. '
        'Use the slider to compare a softer or stronger enhancement.'
    )

    png = cv2.imencode('.png', (enhanced * 255).astype('uint8'))[1].tobytes()
    st.download_button(
        'Download final enhanced image',
        png,
        'enhanced.png',
        'image/png'
    )
