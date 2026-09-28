import os
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from model import NightDehazeNet
from inference import enhance_tiled, safe_quality_guard

st.set_page_config(page_title='NightVision-Dehaze v2', layout='wide')
st.title('NightVision-Dehaze v2')
st.caption('ACDC-trained CNN + Channel Attention + Transformer | full-resolution tiled restoration')

CHECKPOINT = 'checkpoints/acdc_v2_best.pth'


@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(CHECKPOINT):
        return None, device, 256
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model = NightDehazeNet(base=ckpt.get('base', 24)).to(device).float()
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, device, int(ckpt.get('size', 256))


def enhance_postprocess(rgb, strength=0.45):
    """Gentle LAB contrast + sharpening after neural restoration."""
    img8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    contrast = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2RGB)
    blurred = cv2.GaussianBlur(contrast, (0, 0), 1.0)
    sharpened = cv2.addWeighted(contrast, 1.10, blurred, -0.10, 0)
    alpha = float(np.clip(strength, 0.0, 1.0))
    enhanced = cv2.addWeighted(img8, 1.0 - alpha, sharpened, alpha, 0)
    return np.clip(enhanced.astype(np.float32) / 255.0, 0, 1)


file = st.file_uploader(
    'Upload a nighttime / foggy / rainy / snowy road image',
    type=['jpg', 'jpeg', 'png']
)
model, device, tile_size = load_model()

if model is None:
    st.warning('Train the v2 ACDC model first: python train_acdc.py --data "E:\\rgb_anon_trainvaltest\\rgb_anon" --epochs 5 --limit 500 --batch 2 --size 256')
elif file:
    img = np.array(Image.open(file).convert('RGB'))
    original = img.astype(np.float32) / 255.0

    with st.spinner('Running full-resolution tiled restoration...'):
        model_out = enhance_tiled(model, img, device, tile=tile_size, overlap=64)
        guarded, orig_sharp, out_sharp, change, was_guarded = safe_quality_guard(original, model_out)

    st.subheader('Post-processing image enhancer')
    strength = st.slider(
        'Enhancement strength',
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help='Gentle local contrast and sharpening applied after the neural restoration.'
    )
    final = enhance_postprocess(guarded, strength)

    col1, col2, col3 = st.columns(3)
    col1.image(img, caption='1. Original Input', use_container_width=True)
    col2.image(guarded, caption='2. Neural Model Output (quality guarded)', use_container_width=True)
    col3.image(final, caption='3. Final Enhanced Output', use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric('Original sharpness', f'{orig_sharp:.1f}')
    m2.metric('Model sharpness', f'{out_sharp:.1f}')
    m3.metric('Mean pixel change', f'{change:.3f}')

    if was_guarded:
        st.warning('Quality guard activated: the model output changed too much or became substantially softer, so the app blended it toward the original image.')
    else:
        st.success('Quality guard passed: no extreme degradation was detected.')

    st.info('V2 processes the original-resolution image as overlapping 256×256 tiles instead of shrinking the whole image to 128×128. This preserves much more road and object detail.')

    png = cv2.imencode('.png', (final * 255).astype('uint8'))[1].tobytes()
    st.download_button('Download final enhanced image', png, 'enhanced_v2.png', 'image/png')
