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
if "history" not in st.session_state:
    st.session_state.history = []
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
    max-width: 100%;
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
.movie-desc {{
    font-size: 0.95rem;
    color: #CCCCCC;
    margin-top: 0.4rem;
    line-height: 1.5;
    font-style: italic;
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

tab1, tab2, tab3 = st.tabs(["📷 Use Webcam", "📝 Write Review", "📜 History"])

# --------------------------
# 📷 Webcam Layout (LEFT)
# --------------------------
with tab1:
    col1, col2 = st.columns([1, 2])
    new_entry = None  # Track whether a new recommendation was added

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
                if recommendations is not None and not recommendations.empty:
                    new_entry = {
                        "emotion": emotion.capitalize(),
                        "emoji": emoji,
                        "movies": recommendations.to_dict(orient='records')
                    }
                    st.session_state.history.append(new_entry)

with col2:
    if image is not None:
        if recommendations is not None and not recommendations.empty:
            st.markdown("### 🍿 Recommended Movies")
            for idx, row in recommendations.iterrows():
                st.markdown(f"""
                    <div class='recommendation-card'>
                        <img src="{row['Poster_URL']}" class='poster'>
                        <div>
                            <div class='movie-title'>{row['Movie_Title']}</div>
                            <div style='margin-top:10px;'>
                                {''.join(f"<span class='genre-tag'>{g.strip()}</span>" for g in row['Genre'].split(','))}
                            </div>
                            <p class='movie-desc' style="margin-top:12px; color:#DDD; font-size:0.95rem; line-height:1.5;">
                                {row.get('Description', 'No description available.')}
                            </p>
                            <p style="margin-top: 10px;">
                                <a href="{row['YouTube_URL']}" target="_blank" style="
                                    display: inline-block;
                                    padding: 8px 16px;
                                    background: linear-gradient(135deg, #FF4B4B, #FF9F43);
                                    color: white;
                                    font-weight: 600;
                                    border-radius: 30px;
                                    text-decoration: none;
                                    margin-right: 10px;
                                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                                " onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 0 12px #FF9F43';"
                                onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='none';">
                                    ▶️ Watch Trailer
                                </a>
                                <a href="https://www.imdb.com/find?q={row['Movie_Title'].replace(' ', '+')}" target="_blank" style="
                                    display: inline-block;
                                    padding: 8px 16px;
                                    background: linear-gradient(135deg, #FFD166, #FF9F43);
                                    color: black;
                                    font-weight: 600;
                                    border-radius: 30px;
                                    text-decoration: none;
                                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                                " onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 0 12px #FFD166';"
                                onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='none';">
                                    ⭐ View on IMDb
                                </a>
                            </p>
                            <p style="margin-top: 14px;">
                                <span style="
                                    display: inline-block;
                                    padding: 4px 12px;
                                    border-radius: 20px;
                                    background: #f5c518;
                                    color: black;
                                    font-weight: bold;
                                    font-size: 0.9rem;
                                    margin-right: 12px;
                                    box-shadow: 0 0 10px rgba(245, 197, 24, 0.5);
                                ">
                                    ⭐ {row.get('Rating', 'N/A')}/10
                                </span>
                            </p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        else:
            st.warning("No recommendations found. Try a different expression.")



# --------------------------
# 📝 Text Review Layout
# --------------------------
with tab2:
    new_entry = None
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
                    new_entry = {
                        "emotion": emotion.capitalize(),
                        "emoji": emoji,
                        "movies": recommendations.to_dict(orient='records')
                    }
                    st.session_state.history.append(new_entry)

            if recommendations is not None and not recommendations.empty:
                st.markdown("### 🎬 Top Movie Suggestions")
                for idx, row in recommendations.iterrows():
                    like_key = f"text_like_{idx}"
                    dislike_key = f"text_dislike_{idx}"

                    st.markdown(f"""
                        <div class='recommendation-card'>
                            <img src="{row['Poster_URL']}" class='poster'>
                            <div>
                                <div class='movie-title'>{row['Movie_Title']}</div>
                                <div style='margin-top:10px;'>
                                    {''.join(f"<span class='genre-tag'>{g.strip()}</span>" for g in row['Genre'].split(','))}
                                </div>
                                <p class='movie-desc' style="margin-top:12px;">
                                    {row.get('Description', 'No description available.')}
                                </p>
                                <p style="margin-top: 10px;">
                                    <a href="{row['YouTube_URL']}" target="_blank" style="
                                        display: inline-block;
                                        padding: 8px 16px;
                                        background: linear-gradient(135deg, #FF4B4B, #FF9F43);
                                        color: white;
                                        font-weight: 600;
                                        border-radius: 30px;
                                        text-decoration: none;
                                        margin-right: 10px;
                                        transition: transform 0.2s ease, box-shadow 0.2s ease;
                                    " onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 0 12px #FF9F43';"
                                    onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='none';">
                                        ▶️ Watch Trailer
                                    </a>
                                    <a href="https://www.imdb.com/find?q={row['Movie_Title'].replace(' ', '+')}" target="_blank" style="
                                        display: inline-block;
                                        padding: 8px 16px;
                                        background: linear-gradient(135deg, #FFD166, #FF9F43);
                                        color: black;
                                        font-weight: 600;
                                        border-radius: 30px;
                                        text-decoration: none;
                                        transition: transform 0.2s ease, box-shadow 0.2s ease;
                                    " onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 0 12px #FFD166';"
                                    onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='none';">
                                        ⭐ View on IMDb
                                    </a>
                                </p>
                                <p style="margin-top: 14px;">
                                    <span style="
                                        display: inline-block;
                                        padding: 4px 12px;
                                        border-radius: 20px;
                                        background: #f5c518;
                                        color: black;
                                        font-weight: bold;
                                        font-size: 0.9rem;
                                        margin-right: 12px;
                                        box-shadow: 0 0 10px rgba(245, 197, 24, 0.5);
                                    ">
                                        ⭐ {row.get('Rating', 'N/A')}/10
                                    </span>
                                </p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)


                    st.markdown("<hr>", unsafe_allow_html=True)
            else:
                st.warning("No matching movies found.")
        else:
            st.error("Please enter some text to analyze.")


with tab3:
    st.markdown("<h3 style='text-align:center;'>📜 Your Emotion History</h3>", unsafe_allow_html=True)

    if st.button("🧹 Clear History", type="primary"):
        st.session_state.history.clear()
        st.success("Emotion history cleared!")

    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"<h4>{i}. {entry['emotion']} {entry['emoji']}</h4>", unsafe_allow_html=True)
            for movie in entry["movies"]:
                st.markdown(f"""
                    <div class='recommendation-card'>
                        <img src="{movie['Poster_URL']}" class='poster'>
                        <div>
                            <div class='movie-title'>{movie['Movie_Title']}</div>
                            <div style='margin-top:10px;'>
                                {''.join(f"<span class='genre-tag'>{g.strip()}</span>" for g in movie.get('Genre', '').split(','))}
                            </div>
                            <p class='movie-desc' style="margin-top:12px;">
                                {movie.get('Description', 'No description available.')}
                            </p>
                            <p style="margin-top: 10px;">
                                <a href="{movie.get('YouTube_URL', '#')}" target="_blank" style="
                                    display: inline-block;
                                    padding: 8px 16px;
                                    background: linear-gradient(135deg, #FF4B4B, #FF9F43);
                                    color: white;
                                    font-weight: 600;
                                    border-radius: 30px;
                                    text-decoration: none;
                                    margin-right: 10px;
                                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                                " onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 0 12px #FF9F43';"
                                onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='none';">
                                    ▶️ Watch Trailer
                                </a>
                                <a href="https://www.imdb.com/find?q={movie['Movie_Title'].replace(' ', '+')}" target="_blank" style="
                                    display: inline-block;
                                    padding: 8px 16px;
                                    background: linear-gradient(135deg, #FFD166, #FF9F43);
                                    color: black;
                                    font-weight: 600;
                                    border-radius: 30px;
                                    text-decoration: none;
                                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                                " onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 0 12px #FFD166';"
                                onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='none';">
                                    ⭐ View on IMDb
                                </a>
                            </p>
                            <p style="margin-top: 14px;">
                                <span style="
                                    display: inline-block;
                                    padding: 4px 12px;
                                    border-radius: 20px;
                                    background: #f5c518;
                                    color: black;
                                    font-weight: bold;
                                    font-size: 0.9rem;
                                    margin-right: 12px;
                                    box-shadow: 0 0 10px rgba(245, 197, 24, 0.5);
                                ">
                                    ⭐ {movie.get('Rating', 'N/A')}/10
                                </span>
                            </p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
    else:
        st.info("No history yet. Use the other tabs to get movie suggestions first!")


# Footer
st.markdown("""
<hr><div style="text-align:center; font-size:0.9rem; color:#888;">
Made with ❤️ by the Movie Emotion Team | © 2025
</div>
""", unsafe_allow_html=True)
