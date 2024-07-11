import datetime
import io
import os

import torch
import numpy as np
import streamlit as st
import cv2
from PIL import Image

from WB.classes.WBsRGB import WB
from bsrgan import RRDBNet as BSRGAN
from swinir import SwinIR

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def clahe(img, clipLimit=2, tileGridSize=(8, 8)):
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    b_clahe = clahe.apply(b)
    g_clahe = clahe.apply(g)
    r_clahe = clahe.apply(r)
    img_clahe = cv2.merge((b_clahe, g_clahe, r_clahe))
    return img_clahe


def white_balance(LR_img, K = 25, sigma = 0.25):
    upgraded_model = 0
    gamut_mapping = 2
    model = WB(gamut_mapping=gamut_mapping, upgraded=upgraded_model, \
               K=K, sigma=sigma)
    WB_img = model.correctImage(LR_img)
    return (WB_img * 255).astype(np.uint8)


def bsrgan(LR_img):
    weight="weights/bsrgan.pth"
    model = BSRGAN()
    model.load_state_dict(torch.load(weight))
    model.eval()
    model.to(device)
    
    # 0-255 -> 0-1
    LR_img = LR_img * 1.0 / 255
    # HCW-BGR to CHW-RGB
    LR_img =  np.transpose(LR_img[:, :, [2, 1, 0]], (2, 0, 1))
    # CHW-RGB to NCHW-RGB
    LR_img = torch.from_numpy(LR_img).float().unsqueeze(0).to(device)
    # Inference
    with torch.no_grad():
        SR_img = model(LR_img).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    # CHW-RGB to HCW-BGR
    SR_img = np.transpose(SR_img[[2, 1, 0], :, :], (1, 2, 0))
    SR_img = (SR_img * 255).astype(np.uint8)
    return SR_img


def swinir_gan(LR_img):
    model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                        num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler="nearest+conv", resi_connection='1conv')
    param_key_g = 'params_ema'
    window_size = 8
    scale = 4

    pretrained_model = torch.load("weights/swinir_gan.pth")
    model.load_state_dict(pretrained_model[param_key_g], strict=True)
    model.eval()
    model.to(device)

    # 0-255 -> 0-1
    LR_img = LR_img * 1.0 / 255
    # HCW-BGR to CHW-RGB
    LR_img =  np.transpose(LR_img[:, :, [2, 1, 0]], (2, 0, 1))
    # CHW-RGB to NCHW-RGB
    LR_img = torch.from_numpy(LR_img).float().unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, h_old, w_old = LR_img.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        LR_img = torch.cat([LR_img, torch.flip(LR_img, [2])], 2)[:, :, :h_old + h_pad, :]
        LR_img = torch.cat([LR_img, torch.flip(LR_img, [3])], 3)[:, :, :, :w_old + w_pad]
        SR_img = model(LR_img)
        SR_img = SR_img[..., :h_old * scale, :w_old * scale]
        SR_img = SR_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    # CHW-RGB to HCW-BGR
    SR_img = np.transpose(SR_img[[2, 1, 0], :, :], (1, 2, 0))
    SR_img = (SR_img * 255).astype(np.uint8)
    return SR_img


def main():
    st.set_page_config(layout="wide")
    st.title("Enhance color and quality of underwater image ")   

    # Columns options
    col1, col2, col3 = st.columns(3)
    highlight_object = col1.checkbox('Highlight object')
    color_improvement = col2.checkbox('Color improvement')
    if color_improvement:
        options = col2.radio(
        "Choose your option",
        [ 
         "Color improvement - Upscale",
         "Upscale- Color improvement"
         ],
        )


    K = col2.number_input("Input K", 
                          value=50, min_value=1, max_value=1000, step =1, placeholder='Default is 50')
    sigma = col2.number_input("Input sigma", value=0.25)

    option_upscale = col1.selectbox('Choose the model',
                        ('SWINIR-GAN', 'BSRGAN'))

    LR_upload = col1.file_uploader("Upload image...", type=['png', 'jpeg', 'jpg'])

    # Column 3
    enhance_button = col3.button('Enhance')

    # Columns image
    col1_, col2_ = st.columns(2)
    with col1_:
        if LR_upload:
            st.image(LR_upload, use_column_width=True, caption="Input image")

    with col2_:
        if enhance_button:
            if LR_upload:
                # PIL -> RGB -> BGR
                PIL_image = Image.open(io.BytesIO(LR_upload.getvalue()))
                RGB_image = np.array(PIL_image)
                BGR_image = cv2.cvtColor(RGB_image, cv2.COLOR_RGB2BGR)

                with st.spinner('Wait for it...'):
                    # Highlight object
                    if highlight_object:
                        BGR_image = clahe(BGR_image)

                    # Color improvement + SR
                    if color_improvement:
                        # Color improvement -> SR
                        if options == 'Color improvement - Upscale':
                            BGR_image = white_balance(BGR_image, K, sigma)
                            if option_upscale == 'BSRGAN':
                                BGR_image = bsrgan(BGR_image)
                            elif option_upscale == 'SWINIR-GAN':
                                BGR_image = swinir_gan(BGR_image)
                        # SR -> Color improvement
                        elif  options == 'Upscale- Color improvement':
                            if option_upscale == 'BSRGAN':
                                BGR_image = bsrgan(BGR_image)
                            elif option_upscale == 'SWINIR-GAN':
                                BGR_image = swinir_gan(BGR_image)
                            BGR_image = white_balance(BGR_image, K, sigma) 
                    # No Color improvement
                    else:
                        # SR
                        if option_upscale == 'BSRGAN':
                            BGR_image = bsrgan(BGR_image)
                        elif option_upscale == 'SWINIR-GAN':
                            BGR_image = swinir_gan(BGR_image)

                final = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2RGB)
                st.image(final, use_column_width="always",
                        caption='Enhancement image')
                
    # Plot images in save folder
    plot_save_button = col3.button('Plot all save images')
    if plot_save_button:
        folder_path = "save_images"

        # Get all image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png"))]

        # Display the images
        for i in range(0, len(image_files), 3):
            # Create a new row
            row = st.columns(3)

            # Display the 3 images in the current row
            for j in range(3):
                if i + j < len(image_files):
                    image_path = os.path.join(folder_path, image_files[i + j])
                    col = row[j]
                    col.image(image_path, caption=image_files[i + j])
                
    # Define the folder path
    clear_files_button = col3.button('Clear all images in save folder')
    folder_path = "save_images"

    if clear_files_button:
        # Get all files in the folder
        files = os.listdir(folder_path)

        # Loop through each file and delete it
        for file in files:
            if os.path.isfile(os.path.join(folder_path, file)):
                os.remove(os.path.join(folder_path, file))

if __name__ == "__main__":
    main()