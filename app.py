Skip to content
Ariqakh
cnn-gonggong
Repository navigation
Code
Issues
Pull requests
Agents
Actions
Projects
Wiki
Security and quality
Insights
Settings
Files
Go to file
t
T
.devcontainer
README.md
app.py
logo_gonggong.png
logo_umrah.png
model_gonggong.h5
packages.txt
requirements.txt
runtime.txt
cnn-gonggong
/
app.py
in
main

Edit

Preview
Indent mode

Spaces
Indent size

4
Line wrap mode

No wrap
Editing app.py file contents
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageFilter
import numpy as np
import base64
from io import BytesIO

st.set_page_config(
    page_title="Klasifikasi Jenis Gonggong",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    margin: 0;
    padding: 0;
    height: 100%;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #A7FFFF 0%, #D1FFFF 100%) !important;
    min-height: 100vh !important;
}

/* Hilangkan padding bawah bawaan streamlit block yang bikin scroll kosong kepanjangan */
[data-testid="stMain"] {
    background: transparent !important;
    padding-bottom: 20px !important; 
}

Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.
