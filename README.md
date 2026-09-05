### ENTER YOUR NAME: LAKSHANYA.N

### ENTER YOUR REGISTER NO. 212224230136

### DATE: 05-09-2026

# Project Based Experiment

### Objective:

To perform sentiment analysis on Facebook data using Natural Language Processing (NLP) techniques and classify the feedback into Positive, Negative, and Neutral sentiments, and filter the data containing only Neutral feedback.

### Program:

```python
# Install required library
!pip install vaderSentiment

# Import libraries
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Read the Facebook dataset
df = pd.read_csv("pseudo_facebook.csv")

# Display the first five records
print("First 5 Records:")
print(df.head())

# Create feedback based on Facebook activity
def create_feedback(row):
    if row["likes_received"] > 100:
        return "I received many likes on Facebook"
    elif row["likes_received"] < 10:
        return "I received very few likes on Facebook"
    else:
        return "I received an average number of likes on Facebook"

df["Feedback"] = df.apply(create_feedback, axis=1)

# Create sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to classify sentiment
def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))

    if score["compound"] >= 0.05:
        return "Positive"
    elif score["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["Sentiment"] = df["Feedback"].apply(get_sentiment)

# Display sentiment results
print("\nSentiment Results:")
print(df[["Feedback", "Sentiment"]].head(20))

# Count each sentiment
print("\nSentiment Counts:")
print(df["Sentiment"].value_counts())

# Filter only neutral feedback
neutral_data = df[df["Sentiment"] == "Neutral"]

# Display neutral feedback
print("\nNeutral Feedback:")
print(neutral_data[["Feedback", "Sentiment"]].head(20))

# Display number of neutral records
print("\nNumber of Neutral Feedback:", len(neutral_data))

# Save neutral feedback into a new CSV file
neutral_data.to_csv("neutral_facebook_feedback.csv", index=False)

print("\nNeutral feedback saved successfully.")
```

### Output:
<img width="920" height="442" alt="image" src="https://github.com/user-attachments/assets/e19f3051-ff1e-4b7d-a87f-13b8027e2a0f" />
<img width="610" height="470" alt="image" src="https://github.com/user-attachments/assets/ed84fa21-b712-4c22-9be6-cc14d7d5257d" />
<img width="532" height="105" alt="image" src="https://github.com/user-attachments/assets/f674f1a0-eafe-4e4c-8c16-90208b3a8cda" />







<img width="427" height="48" alt="image" src="https://github.com/user-attachments/assets/3ed9eb8c-f239-4175-a441-b239f7199553" />


### Inference:

This project helped me understand how sentiment analysis can be performed on Facebook data using Natural Language Processing techniques. I learned how to use the VADER sentiment analyzer to classify feedback into Positive, Negative, and Neutral categories. I also learned how to filter the dataset to obtain only Neutral feedback and save the filtered data as a separate CSV file. This experiment improved my understanding of data preprocessing, sentiment classification, and data filtering using Python.
