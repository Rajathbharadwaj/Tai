import streamlit as st
from cnn_mnist_pytorch import main as pyt
from cnn_mnist_flax_jax import main as fj

st.title('MNIST Project using Flax and Jax and PyTorch')

option = st.sidebar.selectbox('Select what you want to do', options=['Train', 'View Metrics'])

if option == 'View Metrics':
    st.title('View Metrics using W&B')
    import streamlit.components.v1 as components
    components.html(""" 
    <iframe src="https://wandb.ai/rajathdb/CNN-MNIST-using-Flax-and-Pytorch/reports/Metrics-for-CNN--VmlldzoyODA3ODgx?accessToken=v78vwetcvkb63n6x53lz1eoqy89euaro2by29012d8v10ipoe0goz712hfr4meri" style="border:none;height:1024px;width:100%">
    """, height=1080)
else:
    if st.button('Run with Flax'):
        with st.spinner('Running with Flax...'):
            fj()

    elif st.button('Run with PyTorch'):
        with st.spinner('Running with PyTorch...'):
            pyt()