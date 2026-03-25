import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np

# ── Load clean data ──────────────────────────────────────────────────────────
df = pd.read_excel("Fishing_clean.xlsx")

# ── Encode features ──────────────────────────────────────────────────────────
tidal_map = {'Low': 0, 'Mid': 1, 'High': 2}
df['Tidal_Encoded'] = df['Tidal_Coeff'].map(tidal_map)

le = LabelEncoder()
df['Fish_Encoded'] = le.fit_transform(df['Fish'])

X = df[['Tidal_Encoded']]
y = df['Fish_Encoded']

# ── Model: Constrained Decision Tree with 5-Fold CV ─────────────────────────
dt = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=8,
    class_weight='balanced',
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(dt, X, y, cv=cv, scoring='f1_weighted')

print("=== Model Performance (5-Fold Cross Validation) ===")
print(f"F1 scores per fold: {scores.round(2)}")
print(f"Mean F1: {scores.mean():.2f} (+/- {scores.std():.2f})")
print("\nNote: Low F1 is expected with one feature and 247 samples.")
print("This model is exploratory, not predictive.")

# Fit on full data for visualisation
dt.fit(X, y)

# ── EDA Visualisations ───────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
colors = sns.color_palette("muted")

species_order = ['Thornfish', 'Spotted Grunter', 'Dory Snapper',
                 'Common Ponyfish', 'Cape Stumpnose', 'Blacktail', 'Other']
tidal_order = ['Low', 'Mid', 'High']

# ── Plot 1: Species distribution ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
counts = df['Fish'].value_counts().reindex(species_order)
sns.barplot(x=counts.index, y=counts.values, palette='muted', ax=ax)
ax.set_title('Total Catch by Species', fontsize=14, fontweight='bold')
ax.set_xlabel('Species')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('plot1_species_distribution.png', dpi=150)
plt.show()
print("Saved plot1_species_distribution.png")

# ── Plot 2: Tidal condition vs species (stacked bar) ─────────────────────────
ct = pd.crosstab(df['Tidal_Coeff'], df['Fish'])
ct = ct.reindex(tidal_order)[species_order]

ct.plot(kind='bar', stacked=True, figsize=(10, 6),
        colormap='tab10', edgecolor='white')
plt.title('Species Caught by Tidal Coefficient', fontsize=14, fontweight='bold')
plt.xlabel('Tidal Coefficient')
plt.ylabel('Number of Fish Caught')
plt.xticks(rotation=0)
plt.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('plot2_tidal_vs_species.png', dpi=150)
plt.show()
print("Saved plot2_tidal_vs_species.png")

# ── Plot 3: Quality fish (Grunter + Kob) by tidal condition ──────────────────
quality = ['Spotted Grunter', 'Dusky Kob', 'Javelin Grunter', 'Shad', 'Blacktail']
df_quality = df[df['Fish'].isin(quality)]

fig, ax = plt.subplots(figsize=(8, 5))
quality_ct = pd.crosstab(df_quality['Tidal_Coeff'], df_quality['Fish'])
quality_ct = quality_ct.reindex(tidal_order).fillna(0)
quality_ct.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='white')
ax.set_title('Quality Species by Tidal Coefficient', fontsize=14, fontweight='bold')
ax.set_xlabel('Tidal Coefficient')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=0)
plt.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('plot3_quality_fish_by_tide.png', dpi=150)
plt.show()
print("Saved plot3_quality_fish_by_tide.png")

# ── Plot 4: Bait effectiveness ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bait_counts = df['Bait_Used'].value_counts()
sns.barplot(x=bait_counts.index, y=bait_counts.values, palette='muted', ax=ax)
ax.set_title('Catches by Bait Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Bait')
ax.set_ylabel('Number of Fish Caught')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('plot4_bait_effectiveness.png', dpi=150)
plt.show()
print("Saved plot4_bait_effectiveness.png")

# ── Plot 5: Decision tree ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(dt,
          feature_names=['Tidal Condition'],
          class_names=le.classes_,
          filled=True,
          rounded=True,
          fontsize=10,
          ax=ax)
plt.title('Decision Tree: Tidal Coefficient → Species', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot5_decision_tree.png', dpi=150)
plt.show()
print("Saved plot5_decision_tree.png")

# ── Print key crosstabs ───────────────────────────────────────────────────────
print("\n=== Tidal Condition vs Species (counts) ===")
print(pd.crosstab(df['Tidal_Coeff'], df['Fish']).reindex(tidal_order))

print("\n=== Quality Fish by Tidal Condition ===")
print(quality_ct)

# ── Plot 6: Catches by Tidal Coefficient and Location ───────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

sns.countplot(
    data=df,
    x='Tidal_Coeff',
    hue='Location',
    order=tidal_order,
    palette='muted',
    ax=ax
)

ax.set_title('Catches by Tidal Coefficient and Location',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Tidal Coefficient')
ax.set_ylabel('Number of Fish Caught')
ax.tick_params(axis='x', rotation=0)

plt.legend(title='Location')
plt.tight_layout()
plt.savefig('plot6_tide_vs_location.png', dpi=150)
plt.show()

print("Saved plot6_tide_vs_location.png")

# ── Plot 7: Species vs Tidal Coefficient by Location ─────────────────────────
"""g = sns.catplot(
    data=df,
    x='Tidal_Coeff',
    hue='Fish',
    col='Location',
    kind='count',
    order=tidal_order,
    col_order=['Harbour', 'Blue Lagoon'],
    height=5,
    aspect=1.2,
    palette='tab10'
)

g.set_titles("{col_name}")
g.set_axis_labels("Tidal Coefficient", "Number of Fish Caught")

for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('plot7_species_tide_location.png', dpi=150)
plt.show()

print("Saved plot7_species_tide_location.png")
"""

# ── Plot 8: Spotted Grunter vs Tidal Coefficient by Location ───────────────
#Seeing if what I noticed in fugure 3 was correct

df_grunter = df[df['Fish'] == 'Spotted Grunter'].copy()

fig, ax = plt.subplots(figsize=(10, 6))

sns.countplot(
    data=df_grunter,
    x='Tidal_Coeff',
    hue='Location',
    order=tidal_order,
    palette='Set2',
    ax=ax
)

ax.set_title('Spotted Grunter Catches by Tidal Coefficient and Location',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Tidal Coefficient')
ax.set_ylabel('Number of Grunter Caught')
ax.tick_params(axis='x', rotation=0)

plt.legend(title='Location')
plt.tight_layout()
plt.savefig('plot7_grunter_tide_location.png', dpi=150)
plt.show()

print("Saved plot7_grunter_tide_location.png")