# 🎭 Emotion-Based Movie Recommender System 🎬

This project recommends movies based on the user's **emotion**—detected either through webcam (facial expression analysis using DeepFace) or written reviews (sentiment analysis using TextBlob). It's built with **Streamlit** for the frontend and integrates emotion detection, NLP, and content-based recommendation logic.


## 🚀 Features

- 🎥 **Webcam-Based Emotion Detection** (using `DeepFace`)
- 📝 **Text Sentiment Analysis** (using `TextBlob`)
- 📽️ Recommends Movies with:
  - 🎞️ Poster Image
  - 🏷️ Genre Tags
  - ⭐ IMDb-style Rating
  - 🎬 Trailer and IMDb buttons
- 📜 History tab to track past emotions and recommendations
- 💡 Beautiful dark UI with glowing animations, cards, and responsive design

## 🚀 Setup Instructions

### 1. Clone the Repository

git clone https://github.com/your-username/emotion-movie-recommender.git
cd emotion-movie-recommend

### 2. Create a Virtual Environment
python -m venv .venv
.venv\Scripts\activate

### 3. Install the Required Packages
pip install -r requirements.txt

### 4. Run the Streamlit App
streamlit run app/streamlit_app.py
