import pandas as pd

# Load the original movie CSV from the data folder
df = pd.read_csv("data/movies_with_posters.csv")

# Add a YouTube trailer search URL
df["YouTube_URL"] = df["Movie_Title"].apply(
    lambda title: f"https://www.youtube.com/results?search_query={title.strip().replace(' ', '+')}+trailer"
)

# Save it back into the same folder or overwrite
df.to_csv("data/movies_with_posters.csv", index=False)

print("✅ Trailer links added to data/movies.csv!")
