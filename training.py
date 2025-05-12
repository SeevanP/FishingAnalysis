import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# Load your dataset with specified columns (A to C) and rows (1 to 33)
df = pd.read_excel("Fishing.xlsx", usecols="A:C", nrows=34, header=0)

# Check the first few rows of data to ensure it's loaded correctly
print("Original data:")
print(df)
print(f"Total rows: {len(df)}")

# Clean the Tidal_Coeff column to remove extra spaces and fix case
df['Tidal_Coeff'] = df['Tidal_Coeff'].str.strip().str.lower()

# Check values after cleaning
print("\nUnique values in Tidal_Coeff after cleaning:")
print(df['Tidal_Coeff'].unique())

# Map Tidal_Coeff to numeric values
tidal_map = {'low': 0, 'mid': 1, 'high': 2}
df['Tidal_Coeff'] = df['Tidal_Coeff'].map(tidal_map)

# Drop rows where mapping failed (NaN)
df = df.dropna(subset=['Tidal_Coeff'])

# One-hot encode Bait_Used
df = pd.get_dummies(df, columns=['Bait_Used'])

# Show mapped and encoded data
print("\nData after mapping Tidal_Coeff:")
print(df[['Tidal_Coeff'] + [col for col in df.columns if 'Bait_Used' in col] + ['Fish']].head())

# Features and target
X = df.drop(columns=['Fish'])
y = df['Fish']

# Print fish counts for awareness
print("\nFish class distribution:")
print(y.value_counts())

# Count how many times each fish appears
fish_counts = Counter(y)

# Only keep fish types that appear at least 2 times
valid_fish = [fish for fish, count in fish_counts.items() if count >= 2]
mask = y.isin(valid_fish)

# Filter the data
X = X[mask]
y = y[mask]

# Print fish counts after filtering
print("\nFish class distribution after filtering:")
print(y.value_counts())

# Train/test split with stratification to ensure balanced classes in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=y.unique(), y=y)
class_weight_dict = dict(zip(y.unique(), class_weights))

# Initialize and train model (Random Forest)
model = RandomForestClassifier(class_weight=class_weight_dict, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Print classification report
print("\n--- Prediction Results ---")
print(classification_report(y_test, y_pred))
#Get more data as it is insufficient