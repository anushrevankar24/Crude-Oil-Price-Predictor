
import streamlit as st

PAGE_CONFIG = {
    "page_title": "WTI Crude Price Predictor",
    "page_icon": "✨",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": None
}

CUSTOM_CSS = """
    <style>  
   
    .stButton button {
        background-color: #00308F;
        color: white !important;
        font-size: 50px;
        border-radius: 8px;
        padding: 7px 15px;
        margin-top: 5px;
    }
    .stButton button:hover {
        background-color:   #002D62;
        color: white !important;
    }
   
    .stTextArea, .stSelectbox {
        width: 100%;
    }
    h1 {
         margin-top: 0px;
        margin-bottom: 3px;
    }
    p {
        margin-top: 5px;
        margin-bottom: 10px;
        
    }
     .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #FFF8F0;
        color: #6c757d;
        text-align: center;
        padding: 5px 0;
        font-size: 10px;
    }
    .footer a {
        color: #FF7518;
        text-decoration: none;
        margin: 0 5px;
    }
    .footer a:hover {
        color: #FF5F1F;
    }
    .footer img {
        height: 20px;
        width: 20px;
        margin-right: 3px;
        vertical-align: middle;
    }
   .prediction-container {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    .prediction-text {
        font-size: 18px;
        color: #333;
        margin: 10px 0;
    }
        body {
            background-color: #f0f2f6;
        }
        .main-title {
            font-size: 50px;
            text-align: center;
            color: #00308F;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .sub-title {
            font-size: 30px;
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
         .sub-sub-title {
            font-size: 25px;
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .header {
            font-size: 28px;
            text-align: center;
            color: #0073e6;
            font-weight: bold;
            margin-top: 20px;
        }
        .input-box {
            text-align: center;
            margin-top: 20px;
        }

    </style>
"""
CAPTION_HTML = """
<div class='caption-container'>
    <p class='caption-text'><strong>Caption {i}:</strong> {caption}</p>
</div>
"""    

FOOTER_HTML = """
<div class="footer">
    <p>
        InstaCaptionPro - 
        <a href="https://github.com/anushrevankar24" target="_blank"><img src="./app/static/github-logo.png" alt="GitHub" /></a>
        <a href="https://www.linkedin.com/in/anushrevankar24" target="_blank"><img src="./app/static/linkedin-logo.png" alt="LinkedIn" /></a>
        <a href="https://instagram.com/anushrevankar24?igshid=ZDdkNTZiNTM=" target="_blank"><img src="./app/static/instagram-logo.png" alt="Instagram" /></a>
    </p>
</div>
"""



