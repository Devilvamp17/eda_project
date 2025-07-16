# importing libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Create results folder
os.makedirs("results", exist_ok=True)

# importing data
data = pd.read_csv("data.csv", encoding='unicode_escape')
print(data.info())
print(data.head(5))
print(data.isna().sum())

# dropping unnecessary columns
data.drop(['Status', 'unnamed1', 'User_ID'], axis=1, inplace=True)
data.dropna(inplace=True)

# Gender countplot
plt.figure()
ax = sns.countplot(x='Gender', data=data)
for bars in ax.containers:
    ax.bar_label(bars)
plt.title("Gender Distribution")
plt.savefig("results/gender_countplot.png")
plt.close()

# Amount by gender
temp = data.groupby(['Amount'], as_index=False).sum().sort_values(by='Amount', ascending=False)
plt.figure()
sns.barplot(x='Gender', y='Amount', data=temp)
plt.title("Amount by Gender")
plt.savefig("results/amount_by_gender.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig("results/correlation_matrix.png")
plt.close()

# identify columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
target = 'Amount'

# convert all numeric columns to float
for i in numerical_cols:
    data[i] = data[i].astype(float)

# Count of Zero Values in Numerical Columns
res = []
for i in numerical_cols:
    res.append((i, data[i].value_counts().get(0, 0)))
res = pd.DataFrame(res, columns=['Column', 'Zero_Count'])

plt.figure(figsize=(10, 5))
sns.barplot(x='Column', y='Zero_Count', data=res)
plt.xticks(rotation=90)
plt.title('Count of Zero Values in Numerical Columns')
plt.savefig("results/zero_value_count.png")
plt.close()

# Original Distributions
plt.figure(figsize=(16, len(numerical_cols) * 3))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(len(numerical_cols), 3, i)
    sns.histplot(data[col], kde=True, color='skyblue', edgecolor='black')
    col_skew = skew(data[col].dropna())
    col_kurt = kurtosis(data[col].dropna())
    plt.title(f"Original Distribution: {col}")
    plt.text(0.95, 0.95,
             f"Skew: {col_skew:.2f}\nKurtosis: {col_kurt:.2f}",
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
plt.tight_layout()
plt.savefig("results/original_distributions.png")
plt.close()

# Scaling based on skew
low_cardinality_numerical = [col for col in numerical_cols if data[col].nunique() <= 10]
binary_cols = [col for col in numerical_cols if data[col].nunique() == 2]
true_numerical_cols = [col for col in numerical_cols if col not in low_cardinality_numerical + binary_cols]

for col in true_numerical_cols:
    col_skew = skew(data[col].dropna())
    if abs(col_skew) < 0.5:
        scaler = StandardScaler()
    elif abs(col_skew) <= 1.5:
        scaler = PowerTransformer(method='yeo-johnson')
    else:
        scaler = RobustScaler()
    data[col] = scaler.fit_transform(data[[col]])
    print(f"{col}: Scaled using {scaler.__class__.__name__} (Skew={col_skew:.2f})")

# Scaled Distributions
plt.figure(figsize=(16, len(true_numerical_cols) * 3))
for i, col in enumerate(true_numerical_cols, 1):
    plt.subplot(len(true_numerical_cols), 3, i)
    sns.histplot(data[col], kde=True, color='skyblue', edgecolor='black')
    col_skew = skew(data[col].dropna())
    col_kurt = kurtosis(data[col].dropna())
    plt.title(f"Scaled Distribution: {col}")
    plt.text(0.95, 0.95,
             f"Skew: {col_skew:.2f}\nKurtosis: {col_kurt:.2f}",
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.75, edgecolor='black'))
plt.tight_layout()
plt.savefig("results/scaled_distributions.png")
plt.close()

# Label Encoding
object_cols = data.select_dtypes(include='object').columns.tolist()
label_encoders = {}
for col in object_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le
    print(f"{col}: label-encoded and converted to numeric.")

# Feature Importance
model = RandomForestRegressor()
model.fit(data.drop(columns=target), data[target])

importance = pd.Series(model.feature_importances_, index=data.drop(columns=target).columns)
plt.figure(figsize=(10, 5))
importance.sort_values(ascending=False).plot(kind='bar', title='Feature Importance')
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.close()

# Target Distribution Plot
sns.histplot(data[target], kde=True)
plt.title('Distribution of Target Variable - Amount')
plt.savefig("results/target_distribution.png")
plt.close()

# Boxplots of categorical features vs target
for cat in categorical_cols:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=cat, y=target, data=data)
    plt.title(f'{target} by {cat}')
    plt.savefig(f"results/{cat}_vs_{target}_boxplot.png")
    plt.close()
