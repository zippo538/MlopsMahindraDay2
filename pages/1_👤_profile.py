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
video_url = "https://www.youtube.com/watch?v=D7awk7_vO0k" 
st.video(video_url)
st.write('\n')
st.header("Projects & Accomplishments", divider='violet')
st.write('\n')
st.header("Experience & Qualifications", divider="orange")
st.write(
    """
    ✔️ More than 2 years of proven expertise in deriving actionable insights from data
    
    ✔️ Demonstrated proficiency and hands-on experience in both Python and Excel
    
    ✔️ Sound grasp of statistical principles as well as their practical application 
    
    ✔️ Organized, detail-oriented individual with strong ability to learn new skills
    """
)
