import pandas as pd
import requests
import os
from time import sleep

# TMDB API config
API_KEY = "30897ca5a402d735c70a3b1270ccfacc"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Paths
BASE_DIR = os.path.dirname(__file__)
INPUT_CSV = os.path.join(BASE_DIR, "data", "movies.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "movies_with_posters.csv")

# Load movie dataset
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print("❌ movies.csv not found in data/ folder.")
    exit(1)

# Prepare output list
poster_data = []

# Process each movie
for index, row in df.iterrows():
    title = row.get("Movie_Title") or row.get("title") or ""
    genre = row.get("Genre", "")

    if not title:
        continue

    params = {"api_key": API_KEY, "query": title}
    try:
        response = requests.get(TMDB_SEARCH_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if data["results"]:
            poster_path = data["results"][0].get("poster_path")
            full_url = POSTER_BASE_URL + poster_path if poster_path else ""
            print(f"✅ Found: {title}")
        else:
            full_url = ""
            print(f"❌ Not found: {title}")

        poster_data.append({
            "Movie_Title": title,
            "Genre": genre,
            "Poster_URL": full_url
        })

        sleep(0.2)  # To avoid hitting API rate limits

    except Exception as e:
        print(f"❌ Error fetching poster for '{title}':", e)
        continue

# Save to CSV
poster_df = pd.DataFrame(poster_data)
poster_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Poster URLs saved to: {OUTPUT_CSV}")
print(f"🎉 Total Movies Processed: {len(poster_df)}")
