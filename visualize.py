import streamlit as st
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import torch

option = st.sidebar.selectbox(
    label='Model Analysis',
    index=None,
    options=('train loss vs. validation loss', 'model inference'),
    placeholder='Select an option'
)

# if 'count' not in st.session_state:
#     st.session_state.count = 0

# def display():
#     pair = st.session_state.pairs[st.session_state.count]
#     # st.write(pair)
#     # st.image

# def next():
#     if st.session_state.count + 1 >= len(st.session_state.pairs):
#         st.session_state.count = 0
#     else:
#         st.session_state.count += 1

# def previous():
#     if st.session_state.count > 0:
#         st.session_state.count -= 1

# c1, c2 = st.columns(2)

# with c1:
#     if st.button("⏮️ Previous", on_click=previous):
#         pass

# with c2:
#     if st.button("Next ⏭️", on_click=next):
#         pass

if option == 'model inference':
    c = st.container(border=True)
    mdl = c.selectbox(
        label='Choose a pretrained model',
        options=os.listdir('results/models/'),
        index=None
    )
    files = c.file_uploader('Choose an image', accept_multiple_files=True)

    c1, c2 = c.columns(2)

    for file in files:
        img = Image.open(file)   
        img = img.resize((512, 512), resample=0) 
        img = np.array(img)

        c1.image(img)

        img = transforms.ToTensor()(img)
        img = img[None, :, :, :]

        model = torch.load(os.path.join('results/models/', mdl)).to('cpu')
        pred = model(img)
        pred = pred.detach().numpy().squeeze()
        pred = pred.transpose(1, 2, 0)
        # pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
        # pred[pred < 0.5] = 0
        # pred[pred > 0.5] = 1

        c2.image(pred, clamp=True)

if option == 'train loss vs. validation loss':
    c = st.container(border=True)
    plt = st.selectbox(
        label='Choose a plot',
        index=None,
        options=os.listdir('results/plots/'),
    )
    if plt is not None:
        plot = np.array(Image.open(os.path.join('results/plots', plt)))
        c.image(plot)