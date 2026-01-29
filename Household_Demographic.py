import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Dataset
data = pd.read_excel("Household_Demographic_Dataset.xlsx")

# Cleaning Dataset
def normalize(col):
    return (
        col.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("/", "")
        .replace("(", "")
        .replace(")", "")
    )

normalized_cols = {normalize(col): col for col in data.columns}

print("\nOriginal Columns:")
print(list(data.columns))

def find_col(keyword):
    for norm_col, original_col in normalized_cols.items():
        if keyword in norm_col:
            return original_col
    raise KeyError(f"Column containing '{keyword}' not found")

AGE_COL = find_col("age")
INCOME_COL = find_col("income")
EDU_COL = find_col("education")
FAMILY_COL = find_col("family")
URBAN_COL = find_col("urban")

print("\nMapped Columns:")
print("Age Column:", AGE_COL)
print("Income Column:", INCOME_COL)
print("Education Column:", EDU_COL)
print("Family Size Column:", FAMILY_COL)
print("Urban/Rural Column:", URBAN_COL)

age_mean = data[AGE_COL].mean()
age_median = data[AGE_COL].median()

income_mean = data[INCOME_COL].mean()
income_median = data[INCOME_COL].median()
income_mode = data[INCOME_COL].mode()[0]

print("\nCentral Tendency:")
print("Age Mean:", age_mean)
print("Age Median:", age_median)
print("Income Mean:", income_mean)
print("Income Median:", income_median)
print("Income Mode:", income_mode)

income_range = data[INCOME_COL].max() - data[INCOME_COL].min()
income_std = data[INCOME_COL].std()
income_var = data[INCOME_COL].var()

print("\nDispersion:")
print("Range:", income_range)
print("Std Dev:", income_std)
print("Variance:", income_var)

Q1 = data[INCOME_COL].quantile(0.25)
Q3 = data[INCOME_COL].quantile(0.75)
IQR = Q3 - Q1

print("\nIQR:", IQR)

plt.figure(figsize=(8,5))
sns.histplot(data[INCOME_COL], kde=True)
plt.title("Household Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()

print("\nSkewness:", data[INCOME_COL].skew())
print("Kurtosis:", data[INCOME_COL].kurt())

plt.figure(figsize=(8,5))
sns.boxplot(x=data[EDU_COL], y=data[INCOME_COL])
plt.title("Income by Education Level")
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(x=data[AGE_COL], y=data[INCOME_COL])
plt.title("Age vs Income")
plt.show()

print("\n COMPLETED SUCCESSFULLY")
