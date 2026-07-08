# ============================================================
# DATA PREPROCESSING PRACTICAL ASSIGNMENT
# Mall Customer Dataset
# Covers: Data Understanding, Data Quality, Cleaning,
# Transformation, Reduction, Proximity Measures
# ============================================================

import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# TASK 1: LOAD DATASET
# ------------------------------------------------------------
# https://github.com/swapnilsaurav/Dataset/blob/master/mall_customer_preprocessing_dataset.csv
file_path = "https://raw.githubusercontent.com/swapnilsaurav/Dataset/refs/heads/master/mall_customer_preprocessing_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
display(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Info:")
df.info()

print("\nDescriptive Statistics:")
display(df.describe(include="all"))


# ------------------------------------------------------------
# TASK 2: DATA QUALITY ASSESSMENT
# ------------------------------------------------------------

print("\nMissing Values:")
missing_summary = df.isnull().sum()
display(missing_summary[missing_summary > 0])

print("\nDuplicate Records:", df.duplicated().sum())

print("\nData Types:")
display(df.dtypes)

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric Columns:", numeric_cols)
print("\nCategorical Columns:", categorical_cols)

# Outlier detection using IQR
print("\nOutlier Count using IQR Method:")
outlier_summary = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outlier_summary[col] = len(outliers)

display(pd.DataFrame(outlier_summary.items(), columns=["Column", "Outlier Count"]))


# ------------------------------------------------------------
# TASK 3: DATA CLEANING
# ------------------------------------------------------------

df_clean = df.copy()

# Remove duplicate rows
df_clean = df_clean.drop_duplicates()

# Standardize text columns: trim spaces and convert to title case
for col in categorical_cols:
    df_clean[col] = df_clean[col].astype(str).str.strip()

# Replace textual missing indicators with actual NaN
df_clean.replace(["nan", "NaN", "None", "", "NULL", "null"], np.nan, inplace=True)

# Clean Gender values
gender_map = {
    "M": "Male", "Male": "Male", "male": "Male", "MALE": "Male",
    "F": "Female", "Female": "Female", "female": "Female", "FEMALE": "Female"
}
df_clean["Gender"] = df_clean["Gender"].map(gender_map)

# Clean MembershipTier values
tier_map = {
    "basic": "Basic", "Basic": "Basic", "BASIC": "Basic",
    "silver": "Silver", "Silver": "Silver", "Silvr": "Silver",
    "gold": "Gold", "GOLD": "Gold", "Gold": "Gold",
    "platinum": "Platinum", "PLATINUM": "Platinum", "Platinum": "Platinum"
}
df_clean["MembershipTier"] = df_clean["MembershipTier"].map(tier_map)

# Standardize City, Country, DeviceType, PaymentMethod, EmailProvider
standard_text_cols = ["City", "Country", "DeviceType", "PaymentMethod", "EmailProvider", "PreferredCategory"]

for col in standard_text_cols:
    df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
    df_clean[col].replace("Nan", np.nan, inplace=True)

# Fix common country variants
country_map = {
    "India": "India",
    "Ind": "India",
    "In": "India",
    "Bharat": "India"
}
df_clean["Country"] = df_clean["Country"].map(country_map).fillna(df_clean["Country"])

# Convert dates into proper datetime format
df_clean["JoinDate"] = pd.to_datetime(df_clean["JoinDate"], errors="coerce", dayfirst=False)
df_clean["LastPurchaseDate"] = pd.to_datetime(df_clean["LastPurchaseDate"], errors="coerce", dayfirst=False)

# Handle invalid ages
df_clean.loc[(df_clean["Age"] < 10) | (df_clean["Age"] > 100), "Age"] = np.nan

# Handle invalid ratings
df_clean.loc[
    (df_clean["SatisfactionRating_1_5"] < 1) |
    (df_clean["SatisfactionRating_1_5"] > 5),
    "SatisfactionRating_1_5"
] = np.nan

# Handle invalid spending score
df_clean.loc[
    (df_clean["SpendingScore_1_100"] < 1) |
    (df_clean["SpendingScore_1_100"] > 100),
    "SpendingScore_1_100"
] = np.nan

# Handle negative values in numeric columns where negative is invalid
non_negative_cols = [
    "AnnualIncome_INR", "AvgBasketValue_INR", "TotalPurchases",
    "OnlinePurchases", "StorePurchases", "LoyaltyPoints"
]

for col in non_negative_cols:
    df_clean.loc[df_clean[col] < 0, col] = np.nan

# Impute numeric missing values with median
numeric_cols_clean = df_clean.select_dtypes(include=["int64", "float64"]).columns.tolist()

for col in numeric_cols_clean:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Impute categorical missing values with mode
categorical_cols_clean = df_clean.select_dtypes(include=["object"]).columns.tolist()

for col in categorical_cols_clean:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Impute dates with median date
for col in ["JoinDate", "LastPurchaseDate"]:
    median_date = df_clean[col].dropna().median()
    df_clean[col] = df_clean[col].fillna(median_date)

print("\nCleaned Dataset Shape:", df_clean.shape)
print("\nMissing Values After Cleaning:")
display(df_clean.isnull().sum())


# ------------------------------------------------------------
# TASK 4: OUTLIER TREATMENT
# ------------------------------------------------------------

df_outlier_treated = df_clean.copy()

# Cap outliers using IQR capping
for col in numeric_cols_clean:
    Q1 = df_outlier_treated[col].quantile(0.25)
    Q3 = df_outlier_treated[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_outlier_treated[col] = np.where(
        df_outlier_treated[col] < lower,
        lower,
        df_outlier_treated[col]
    )

    df_outlier_treated[col] = np.where(
        df_outlier_treated[col] > upper,
        upper,
        df_outlier_treated[col]
    )

print("\nOutlier Treatment Completed.")


# ------------------------------------------------------------
# TASK 5: FEATURE ENGINEERING
# ------------------------------------------------------------

df_featured = df_outlier_treated.copy()

# Create customer tenure in days
df_featured["CustomerTenureDays"] = (
    df_featured["LastPurchaseDate"] - df_featured["JoinDate"]
).dt.days

df_featured["CustomerTenureDays"] = df_featured["CustomerTenureDays"].abs()

# Create purchase ratio features
df_featured["OnlinePurchaseRatio"] = (
    df_featured["OnlinePurchases"] / (df_featured["TotalPurchases"] + 1)
)

df_featured["StorePurchaseRatio"] = (
    df_featured["StorePurchases"] / (df_featured["TotalPurchases"] + 1)
)

# Create value per purchase
df_featured["ValuePerPurchase"] = (
    df_featured["AvgBasketValue_INR"] / (df_featured["TotalPurchases"] + 1)
)

print("\nFeature Engineering Completed.")
display(df_featured.head())


# ------------------------------------------------------------
# TASK 6: DATA TRANSFORMATION
# ------------------------------------------------------------

# Select numeric columns for scaling
numeric_features = df_featured.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Remove target variable from transformation if present
target_col = "ChurnNextMonth"

if target_col in numeric_features:
    numeric_features.remove(target_col)

# Min-Max Normalization
minmax_scaler = MinMaxScaler()
df_minmax = df_featured.copy()
df_minmax[numeric_features] = minmax_scaler.fit_transform(df_minmax[numeric_features])

print("\nMin-Max Normalization Completed.")

# Standardization
standard_scaler = StandardScaler()
df_standard = df_featured.copy()
df_standard[numeric_features] = standard_scaler.fit_transform(df_standard[numeric_features])

print("Standardization Completed.")

# Robust Scaling
robust_scaler = RobustScaler()
df_robust = df_featured.copy()
df_robust[numeric_features] = robust_scaler.fit_transform(df_robust[numeric_features])

print("Robust Scaling Completed.")

print("\nBefore Transformation:")
display(df_featured[numeric_features].head())

print("\nAfter Standardization:")
display(df_standard[numeric_features].head())


# ------------------------------------------------------------
# TASK 7: ENCODING CATEGORICAL VARIABLES
# ------------------------------------------------------------

df_encoded = df_standard.copy()

# Drop date columns and ID/name columns before modelling
drop_cols = ["CustomerID", "CustomerName", "JoinDate", "LastPurchaseDate"]

df_encoded = df_encoded.drop(columns=drop_cols, errors="ignore")

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df_encoded, drop_first=True)

print("\nEncoded Dataset Shape:", df_encoded.shape)
display(df_encoded.head())


# ------------------------------------------------------------
# TASK 8: DATA REDUCTION
# ------------------------------------------------------------

# Separate features and target
X = df_encoded.drop(columns=[target_col], errors="ignore")
y = df_encoded[target_col] if target_col in df_encoded.columns else None

print("\nOriginal Feature Count:", X.shape[1])

# 8.1 Remove low variance features
selector = VarianceThreshold(threshold=0.01)
X_variance = selector.fit_transform(X)

selected_columns = X.columns[selector.get_support()]
X_reduced_variance = pd.DataFrame(X_variance, columns=selected_columns)

print("Feature Count After Variance Threshold:", X_reduced_variance.shape[1])

# 8.2 Correlation-based feature removal
corr_matrix = X_reduced_variance.corr().abs()

upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_features = [
    column for column in upper_triangle.columns
    if any(upper_triangle[column] > 0.90)
]

X_corr_reduced = X_reduced_variance.drop(columns=high_corr_features)

print("Highly Correlated Features Removed:", len(high_corr_features))
print("Feature Count After Correlation Reduction:", X_corr_reduced.shape[1])

# 8.3 PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_corr_reduced)

print("PCA Components Retaining 95% Variance:", X_pca.shape[1])


# ------------------------------------------------------------
# TASK 9: SAMPLING
# ------------------------------------------------------------

# Random sample of 500 records
sample_df = df_featured.sample(n=500, random_state=42)

print("\nSample Dataset Shape:", sample_df.shape)


# ------------------------------------------------------------
# TASK 10: PROXIMITY MEASURES
# ------------------------------------------------------------

# Select 20 records for proximity analysis
proximity_data = X_corr_reduced.head(20)

# Euclidean distance
euclidean_matrix = euclidean_distances(proximity_data)

# Manhattan distance
manhattan_matrix = manhattan_distances(proximity_data)

# Cosine similarity
cosine_matrix = cosine_similarity(proximity_data)

print("\nEuclidean Distance Matrix:")
display(pd.DataFrame(euclidean_matrix))

print("\nManhattan Distance Matrix:")
display(pd.DataFrame(manhattan_matrix))

print("\nCosine Similarity Matrix:")
display(pd.DataFrame(cosine_matrix))

# Find most similar pair using cosine similarity
cosine_df = pd.DataFrame(cosine_matrix)

np.fill_diagonal(cosine_df.values, -1)

most_similar_pair = np.unravel_index(
    np.argmax(cosine_df.values),
    cosine_df.shape
)

print("\nMost Similar Records Based on Cosine Similarity:")
print("Record Index Pair:", most_similar_pair)
print("Similarity Score:", cosine_df.iloc[most_similar_pair])


# ------------------------------------------------------------
# TASK 11: TRAIN-TEST SPLIT OPTIONAL
# ------------------------------------------------------------

if y is not None:
    X_train, X_test, y_train, y_test = train_test_split(
        X_corr_reduced,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)


# ------------------------------------------------------------
# TASK 12: SAVE CLEANED DATASETS
# ------------------------------------------------------------

df_featured.to_csv("mall_customer_cleaned_featured_dataset.csv", index=False)
df_encoded.to_csv("mall_customer_encoded_dataset.csv", index=False)
X_corr_reduced.to_csv("mall_customer_reduced_features.csv", index=False)

print("\nFiles Saved Successfully:")
print("1. mall_customer_cleaned_featured_dataset.csv")
print("2. mall_customer_encoded_dataset.csv")
print("3. mall_customer_reduced_features.csv")