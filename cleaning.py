import pandas as pd

# Load
df = pd.read_excel("Fishing.xlsx", usecols="A:D", header=0)
df.columns = ['Fish', 'Tidal_Coeff', 'Bait_Used','Location']

# Strip whitespace from all string columns
df['Fish'] = df['Fish'].str.strip()
df['Tidal_Coeff'] = df['Tidal_Coeff'].str.strip()
df['Bait_Used'] = df['Bait_Used'].str.strip()
df['Location'] = df['Location'].str.strip()

# Fix known typos
df['Fish'] = df['Fish'].replace({
    'Dusly Kob': 'Dusky Kob',
    'Dusky kob': 'Dusky Kob',   # standardise capitalisation
    'Backtail': 'Blacktail',
    'Grunter': 'Spotted Grunter',
})

# Merge Very Low into Low
df['Tidal_Coeff'] = df['Tidal_Coeff'].replace({'Very Low': 'Low'})

# Group rare species into Other (less than 10 samples)
min_samples = 10
species_counts = df['Fish'].value_counts()
valid_species = species_counts[species_counts >= min_samples].index
df['Fish'] = df['Fish'].apply(lambda x: x if x in valid_species else 'Other')

df['Bait_Used'] = df['Bait_Used'].replace({
    'S': 'Squid',
})

# Standardise tidal coefficient capitalisation
df['Tidal_Coeff'] = df['Tidal_Coeff'].str.title()  # Low, Mid, High, Very Low

# Drop any completely empty rows
df = df.dropna(how='all')

# Check for remaining oddities
print("=== Fish species and counts ===")
print(df['Fish'].value_counts())

print("\n=== Tidal categories ===")
print(df['Tidal_Coeff'].value_counts())

print("\n=== Bait types ===")
print(df['Bait_Used'].value_counts())

print("\n=== Locations ===")
print(df['Location'].value_counts())

print("\n=== Any remaining nulls ===")
print(df.isnull().sum())

print(f"\nTotal clean rows: {len(df)}")

# Save cleaned version
df.to_excel("Fishing_clean.xlsx", index=False)
print("\nSaved to Fishing_clean.xlsx")