from pathlib import Path
import streamlit as st
from PIL import Image
from config.config import Config
import requests
from utils.styling import load_css



st.set_page_config(page_title="Profile", page_icon=Config.PAGE_ICON)


# Fetch CSS content from URL
load_css()


with open(Config.RESUME_PATH, "rb") as pdf_file:
    PDFbyte = pdf_file.read()
profile_pic = Image.open(Config.PROFILE_PATH)
col1, col2 = st.columns(2, gap="large")
with col1:
    st.image(profile_pic)
with col2:
    st.header(Config.NAME, divider='rainbow')
    st.write(":wave: Hi! I'm Mahindra, a Data Scientist from Batang.")
    st.write(Config.DESCRIPTION)
    st.divider()
    st.download_button(
        label=" 📄 Download Resume",
        data=PDFbyte,
        file_name="CV Mahindra",
        mime="application/octet-stream",
    )
    st.write("📫", Config.EMAIL)
    st.write("📱", Config.PHONE_NUMBER)
st.header("Social Links:", divider='red')
cols = st.columns(len(Config.SOCIAL_MEDIA))
for index, (platform, link) in enumerate(Config.SOCIAL_MEDIA.items()):
    cols[index].write(f"[{platform}]({link})")

st.write('\n')
st.header("Experience & Qualifications", divider="orange")
st.write(
    """
    ✔️ IT Perbankan, Digital Talent Scholarship Kominfo
    
    ✔️ Project Based Internship Program : ID/X Partners Data Scientist 
    
    ✔️ Cloud Practitioner Essentials (Belajar Dasar AWS Cloud) (Dicoding)
    
    ✔️ Belajar Membuat Aplikasi Back-End untuk Pemula (Dicoding)
    """
)

st.write('\n')
st.header("Languange", divider="blue")
st.write(
    """
    ✔️ English (Limited Working)
    
    ✔️ Indonesia (Native)
    
    ✔️ German A2
    
    """
)