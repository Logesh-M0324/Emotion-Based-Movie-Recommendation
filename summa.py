import streamlit as st
import pandas as pd
from textblob import TextBlob
import cv2
from deepface import DeepFace
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from recommender.recommend import recommend_movies

# Emojis for detected emotion
emotion_emoji_map = {
    "happy": "😊", "sad": "😢", "angry": "😠",
    "surprise": "😲", "fear": "😨", "neutral": "😐"
}

# Theme colors
COLORS = {
    "primary": "#FF4B4B", "secondary": "#FF9F43", "accent": "#FFD166",
    "background": "#0E1117", "card": "#1E222A", "text": "#FFFFFF"
}

# Page setup
st.set_page_config(page_title="🎬 Emotion Recommender", layout="wide", initial_sidebar_state="collapsed")

# Inject CSS
st.markdown(f"""
<style>
/* ---------- Global Base Styles ---------- */
body {{
    background-color: {COLORS['background']};
    color: {COLORS['text']};
    font-family: 'Montserrat', sans-serif;
    transition: background 0.3s ease, color 0.3s ease;
    overflow-x: hidden;
}}

/* ---------- Stylish Divider ---------- */
hr {{
    border: none;
    height: 4px;
    margin: 2.5rem 0;
    background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['secondary']});
    border-radius: 100px;
    box-shadow: 0 0 10px {COLORS['primary']}, 0 0 14px {COLORS['secondary']};
}}

/* ---------- Hero Heading with Pulse Glow ---------- */
h1 {{
    font-size: 4rem;
    text-align: center;
    font-weight: 900;
    background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['secondary']});
    -webkit-background-clip: text;
    color: transparent;
    position: relative;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 2rem;
    animation: glowPulse 3s infinite ease-in-out;
}}

@keyframes glowPulse {{
    0%, 100% {{ text-shadow: 0 0 12px {COLORS['primary']}, 0 0 20px {COLORS['secondary']}; }}
    50% {{ text-shadow: 0 0 24px {COLORS['secondary']}, 0 0 40px {COLORS['primary']}; }}
}}

h1::after {{
    content: '';
    position: absolute;
    bottom: -16px;
    left: 50%;
    transform: translateX(-50%);
    width: 160px;
    height: 5px;
    border-radius: 4px;
    background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['secondary']});
    animation: underlineWave 2.5s ease-in-out infinite;
}}

@keyframes underlineWave {{
    0% {{ transform: translateX(-50%) scaleX(1); opacity: 0.8; }}
    50% {{ transform: translateX(-50%) scaleX(1.3); opacity: 1; }}
    100% {{ transform: translateX(-50%) scaleX(1); opacity: 0.8; }}
}}

h4 {{
    text-align: center;
    color: #CCCCCC;
    font-size: 1.3rem;
    margin-bottom: 2rem;
    animation: fadeInUp 1.4s ease forwards;
}}

@keyframes fadeInUp {{
    0% {{ opacity: 0; transform: translateY(12px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

/* ---------- Recommendation Cards ---------- */
.recommendation-card {{
    display: flex;
    align-items: flex-start;
    gap: 24px;
    padding: 24px;
    margin-bottom: 2rem;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 12px 45px rgba(0, 0, 0, 0.4);
    transition: transform 0.4s ease, box-shadow 0.4s ease, border 0.4s ease;
    position: relative;
}}
.recommendation-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    justify-content: center;
    margin-top: 2rem;
}}

.recommendation-card {{
    flex: 1 1 300px;
    max-width: 380px;
}}
.recommendation-card:hover {{
    transform: translateY(-10px) scale(1.015);
    box-shadow: 0 30px 60px rgba(255, 215, 102, 0.25), 0 0 20px rgba(255, 107, 107, 0.4);
    border-color: {COLORS['accent']};
}}

/* ---------- Poster Styling ---------- */
.poster {{
    width: 150px;
    height: 220px;
    border-radius: 16px;
    object-fit: cover;
    box-shadow: 0 10px 24px rgba(0,0,0,0.5);
    transition: transform 0.35s ease-in-out;
}}

.recommendation-card:hover .poster {{
    transform: scale(1.08) rotateZ(1deg);
}}

/* ---------- Movie Info Styling ---------- */
.movie-title {{
    color: {COLORS['accent']};
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 0.6rem;
    text-shadow: 0 2px 10px rgba(255, 215, 102, 0.3);
}}

.genre-tag {{
    display: inline-block;
    margin: 6px 8px 0 0;
    padding: 6px 16px;
    border-radius: 30px;
    background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
    color: white;
    font-weight: 600;
    font-size: 0.85rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}

.genre-tag:hover {{
    transform: scale(1.12);
    box-shadow: 0 0 10px {COLORS['secondary']};
}}

/* ---------- Floating Emoji Animation ---------- */
.emoji-float {{
    animation: float 3s ease-in-out infinite;
    display: inline-block;
}}

@keyframes float {{
    0% {{ transform: translateY(0); }}
    50% {{ transform: translateY(-10px); }}
    100% {{ transform: translateY(0); }}
}}
</style>
""", unsafe_allow_html=True)




# Header
st.markdown("""
<h1>🎭 Emotion Movie Recommender 🎬</h1>
<br/>
<h4 style='text-align:center; color:#BBBBBB;'>Find the best movie for your mood <span class='emoji-float'>✨</span></h4>
<hr>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📷 Use Webcam", "📝 Write Review"])

# --------------------------
# 📷 Webcam Layout (LEFT)
# --------------------------
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        image = st.camera_input("Snap a quick selfie to detect your mood 🎥")
        if image is not None:
            with st.spinner("Analyzing facial expression..."):
                file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                emoji = emotion_emoji_map.get(emotion.lower(), "😐")
            st.success(f"Detected Emotion: **{emotion.capitalize()}** {emoji}")

            with st.spinner("Fetching movie suggestions..."):
                recommendations = recommend_movies(emotion)

    with col2:
        if image is not None:
            if recommendations is not None and not recommendations.empty:
                st.markdown("### 🍿 Recommended Movies")
                for _, row in recommendations.iterrows():
                    st.markdown(f"""
                    <div class='recommendation-card'>
                        <img src="{row['Poster_URL']}" class='poster'>
                        <div>
                            <div class='movie-title'>{row['Movie_Title']}</div>
                            <div style='margin-top:10px;'>
                                {''.join(f"<span class='genre-tag'>{g.strip()}</span>" for g in row['Genre'].split(','))}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. Try a different expression.")

# --------------------------
# 📝 Text Review Layout
# --------------------------
with tab2:
    review = st.text_area("Write how you feel or your review:", placeholder="e.g., I feel so inspired today after that movie!")
    if st.button("Analyze Sentiment"):
        if review.strip():
            with st.spinner("Analyzing sentiment..."):
                blob = TextBlob(review)
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    emotion = "happy"
                elif polarity < -0.1:
                    emotion = "sad"
                else:
                    emotion = "neutral"
                emoji = emotion_emoji_map.get(emotion, "😐")
            st.success(f"Detected Sentiment: **{emotion.capitalize()}** {emoji}")

            with st.spinner("Finding movie recommendations..."):
                recommendations = recommend_movies(emotion)

            if recommendations is not None and not recommendations.empty:
                st.markdown("### 🎬 Top Movie Suggestions")
                for _, row in recommendations.iterrows():
                    st.markdown(f"""
                    <div class='recommendation-card'>
                        <img src="{row['Poster_URL']}" class='poster'>
                        <div>
                            <div class='movie-title'>{row['Movie_Title']}</div>
                            <div style='margin-top:10px;'>
                                {''.join(f"<span class='genre-tag'>{g.strip()}</span>" for g in row['Genre'].split(','))}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No matching movies found.")
        else:
            st.error("Please enter some text to analyze.")

# Footer
st.markdown("""
<hr><div style="text-align:center; font-size:0.9rem; color:#888;">
Made with ❤️ by the Movie Emotion Team | © 2025
</div>
""", unsafe_allow_html=True)
