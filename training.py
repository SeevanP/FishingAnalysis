import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# Load your dataset with specified columns (A to C) and rows (1 to 33)
df = pd.read_excel("Fishing.xlsx", usecols="A:C", nrows=34, header=0)

# Check the first few rows of data to ensure it's loaded correctly
print("Original data:")
print(df.head())

# Check unique values in the Tidal_Coeff column to investigate discrepancies
print("\nUnique values in Tidal_Coeff:")
print(df['Tidal_Coeff'].unique())

# Clean the Tidal_Coeff column to remove any extra spaces or inconsistent capitalization
df['Tidal_Coeff'] = df['Tidal_Coeff'].str.strip().str.lower()

# Check the values again after cleaning
print("\nUnique values in Tidal_Coeff after cleaning:")
print(df['Tidal_Coeff'].unique())

# Map Tidal_Coeff to numerical values
tidal_map = {
    'low': 0,
    'mid': 1,
    'high': 2
}

# Apply the mapping to the Tidal_Coeff column, if a value cannot be mapped, it will be replaced with NaN
df['Tidal_Coeff'] = df['Tidal_Coeff'].map(tidal_map)

# Check if mapping worked correctly
print("\nData after mapping Tidal_Coeff:")
print(df[['Tidal_Coeff', 'Bait_Used', 'Fish']].head())

# Handle any rows where Tidal_Coeff is NaN (if needed, you can drop or impute)
df = df.dropna(subset=['Tidal_Coeff'])  # Dropping rows where Tidal_Coeff is NaN

# One-hot encode Bait_Used
df = pd.get_dummies(df, columns=['Bait_Used'])

# Features and target
X = df[['Tidal_Coeff'] + [col for col in df.columns if 'Bait_Used' in col]]  # Include all one-hot encoded bait columns
y = df['Fish']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=y.unique(), y=y)
class_weight_dict = dict(zip(y.unique(), class_weights))

# Model (Logistic Regression for a simpler approach)
model = LogisticRegression(max_iter=200, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Print classification report for evaluation
print("\n--- Prediction Results ---")
print(classification_report(y_test, y_pred))
