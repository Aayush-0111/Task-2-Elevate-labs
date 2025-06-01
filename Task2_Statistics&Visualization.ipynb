# titanic_eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# 1. Generate summary statistics
print("\n--- Summary Statistics ---\n")
print(df.describe(include='all'))
print("\n--- Median Values ---\n")
print(df.median(numeric_only=True))

# 2. Histograms and Boxplots for numeric features
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Histograms
df[numeric_cols].hist(figsize=(12, 10), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Numeric Features")
plt.tight_layout()
plt.show()

# Boxplots
plt.figure(figsize=(15, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

# 3. Correlation Matrix & Pairplot
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Pairplot (only a few important features to avoid clutter)
sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# 4. Patterns / Trends / Anomalies
print("\n--- Missing Values ---\n")
print(df.isnull().sum())

print("\n--- Survival Rate by Class ---\n")
print(df.groupby("Pclass")["Survived"].mean())

print("\n--- Survival Rate by Sex ---\n")
print(df.groupby("Sex")["Survived"].mean())

# 5. Feature-level Inference Notes:
# - Higher class passengers had higher survival rates
# - Women survived at a much higher rate than men
# - Fare distribution has strong outliers (very high fare values)
# - Age has some missing values and outliers
# - Correlation matrix shows 'Fare' is positively correlated with 'Pclass'
