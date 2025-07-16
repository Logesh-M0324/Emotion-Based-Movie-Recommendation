from textblob import TextBlob
import pandas as pd

# Load sample movie data
df = pd.read_csv('../data/movies.csv')  # Ensure the file exists with review text

# Let's say there's a 'Review' column
def get_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Apply to each review
df['Sentiment'] = df['Review'].apply(get_sentiment)

# Show some results
print(df[['Movie_Title', 'Review', 'Sentiment']].head())

# Optional: Save the results
df.to_csv('sentiment_output.csv', index=False)
