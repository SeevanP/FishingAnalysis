import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_excel("Fishing.xlsx")

# Encode features
X = data[["Tidal_Coeff", "Bait_Used"]]
y = data["Fish"]

encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X.loc[:, col] = le.fit_transform(X[col])
    encoders[col] = le

fish_encoder = LabelEncoder()
y = fish_encoder.fit_transform(y)

# Train Naive Bayes
model = CategoricalNB()
model.fit(X, y)


# Function to predict top N fish
def predict_top_fish(sample_dict, top_n=3):
    sample = pd.DataFrame(sample_dict)

    # Encode sample features
    for col in sample.columns:
        sample[col] = encoders[col].transform(sample[col])

    # Get probabilities
    probs = model.predict_proba(sample)[0]

    # Get top N indices
    top_indices = probs.argsort()[::-1][:top_n]

    # Map to fish names and probabilities
    top_fish = [(fish_encoder.inverse_transform([i])[0], probs[i]) for i in top_indices]
    return top_fish


# Example usage
sample_input = {
    "Tidal_Coeff": ["High"],
    "Bait_Used": ["Squid"]
}

top3 = predict_top_fish(sample_input, top_n=3)
print("Top 3 likely fish for this setup:")
for fish, prob in top3:
    print(f"{fish}: {prob:.2%}")
