"""
Data Preprocessing Assignment Solution: 
End-to-End Data Preprocessing for House Price Prediction

Input file expected in the same folder:
    HousePrice_Dataset.csv

Output files:
    HousePrice_Preprocessed.csv
    HousePrice_DataQuality_Report.csv
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ------------------------------------------------------------
# Task 1: Load and explore the dataset
# ------------------------------------------------------------

CSV_PATH = "HousePrice_Dataset.csv"
df = pd.read_csv(CSV_PATH)

print("Initial shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary statistics for raw data:")
print(df.describe(include="all"))

# ------------------------------------------------------------
# Task 2: Data quality assessment
# ------------------------------------------------------------

quality_report = []
for col in df.columns:
    quality_report.append({
        "Column": col,
        "Data Type": str(df[col].dtype),
        "Missing Values": int(df[col].isna().sum()),
        "Unique Values": int(df[col].nunique(dropna=True)),
        "Duplicate Column Values": int(df[col].duplicated().sum())
    })

quality_df = pd.DataFrame(quality_report)
quality_df.to_csv("HousePrice_DataQuality_Report.csv", index=False)
print("\nData Quality Report:")
print(quality_df)

# ------------------------------------------------------------
# Task 3: Data cleaning
# ------------------------------------------------------------

clean_df = df.copy()

# 3.1 Remove complete duplicate rows
clean_df = clean_df.drop_duplicates()

# 3.2 Standardize text columns: remove extra spaces and convert to title case
text_columns = ["Location", "Garage", "Parking", "Furnishing", "PropertyType"]
for col in text_columns:
    clean_df[col] = clean_df[col].astype("string").str.strip()

clean_df["Location"] = clean_df["Location"].str.title()
clean_df["PropertyType"] = clean_df["PropertyType"].str.title()
clean_df["Parking"] = clean_df["Parking"].str.title()

# 3.3 Standardize Garage values
clean_df["Garage"] = clean_df["Garage"].str.lower().replace({
    "yes": "Yes", "y": "Yes",
    "no": "No", "n": "No"
})

# 3.4 Standardize Furnishing values
clean_df["Furnishing"] = clean_df["Furnishing"].str.lower().str.replace("-", " ", regex=False).str.strip()
clean_df["Furnishing"] = clean_df["Furnishing"].replace({
    "furnished": "Furnished",
    "semi furnished": "Semi-Furnished",
    "unfurnished": "Unfurnished"
})

# 3.5 Convert numeric columns stored as text into numeric values
numeric_columns = ["Area", "Bedrooms", "Bathrooms", "Floors", "HouseAge", "DistanceToCity", "Price"]

# Fix text values in Bedrooms before numeric conversion
clean_df["Bedrooms"] = clean_df["Bedrooms"].astype("string").str.strip().replace({
    "2 BHK": "2",
    "Three": "3",
    "three": "3"
})

for col in numeric_columns:
    clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

# 3.6 Treat invalid numerical values as missing
clean_df.loc[clean_df["Area"] <= 0, "Area"] = np.nan
clean_df.loc[(clean_df["Bedrooms"] <= 0) | (clean_df["Bedrooms"] > 10), "Bedrooms"] = np.nan
clean_df.loc[(clean_df["Bathrooms"] <= 0) | (clean_df["Bathrooms"] > 10), "Bathrooms"] = np.nan
clean_df.loc[(clean_df["Floors"] <= 0) | (clean_df["Floors"] > 50), "Floors"] = np.nan
clean_df.loc[(clean_df["HouseAge"] < 0) | (clean_df["HouseAge"] > 100), "HouseAge"] = np.nan
clean_df.loc[(clean_df["DistanceToCity"] < 0) | (clean_df["DistanceToCity"] > 100), "DistanceToCity"] = np.nan
clean_df.loc[clean_df["Price"] <= 0, "Price"] = np.nan

# 3.7 Drop rows where target variable Price is missing
clean_df = clean_df.dropna(subset=["Price"])

# 3.8 Impute missing values
num_features_for_imputation = ["Area", "Bedrooms", "Bathrooms", "Floors", "HouseAge", "DistanceToCity"]
cat_features_for_imputation = ["Location", "Garage", "Parking", "Furnishing", "PropertyType"]

for col in num_features_for_imputation:
    clean_df[col] = clean_df[col].fillna(clean_df[col].median())

for col in cat_features_for_imputation:
    clean_df[col] = clean_df[col].fillna(clean_df[col].mode()[0])

# ------------------------------------------------------------
# Task 4: Outlier treatment and transformation
# ------------------------------------------------------------

def winsorize_iqr(dataframe, column):
    """Cap outliers using the IQR lower and upper bounds."""
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    dataframe[column] = dataframe[column].clip(lower=lower, upper=upper)
    return lower, upper

outlier_columns = ["Area", "Price", "DistanceToCity"]
outlier_bounds = {}
for col in outlier_columns:
    lower, upper = winsorize_iqr(clean_df, col)
    outlier_bounds[col] = {"Lower Bound": lower, "Upper Bound": upper}

print("\nIQR outlier bounds used for winsorization:")
print(pd.DataFrame(outlier_bounds).T)

# Log transformation is useful for positively skewed values such as Area and Price.
clean_df["Log_Area"] = np.log1p(clean_df["Area"])
clean_df["Log_Price"] = np.log1p(clean_df["Price"])

# Binning examples
clean_df["HouseAgeCategory"] = pd.cut(
    clean_df["HouseAge"],
    bins=[-1, 5, 15, 30, 100],
    labels=["New", "Moderate", "Old", "Very Old"]
)

clean_df["DistanceCategory"] = pd.cut(
    clean_df["DistanceToCity"],
    bins=[-1, 5, 15, 100],
    labels=["Near", "Medium", "Far"]
)

# ------------------------------------------------------------
# Task 5: Feature engineering
# ------------------------------------------------------------

clean_df["TotalRooms"] = clean_df["Bedrooms"] + clean_df["Bathrooms"]
clean_df["PricePerSqFt"] = clean_df["Price"] / clean_df["Area"]
clean_df["AreaPerRoom"] = clean_df["Area"] / clean_df["TotalRooms"].replace(0, np.nan)
clean_df["LuxuryHouseFlag"] = np.where(
    (clean_df["Area"] >= clean_df["Area"].quantile(0.75)) &
    (clean_df["Furnishing"] == "Furnished") &
    (clean_df["Parking"].isin(["Covered", "Basement"])),
    "Yes",
    "No"
)
clean_df["PropertySizeCategory"] = pd.cut(
    clean_df["Area"],
    bins=[0, 1000, 2000, 3500, np.inf],
    labels=["Small", "Medium", "Large", "Very Large"]
)
clean_df["PremiumLocationFlag"] = np.where(
    clean_df["Location"].isin(["Banjara Hills", "Jubilee Hills", "Hitech City", "Gachibowli"]),
    "Yes",
    "No"
)

# ------------------------------------------------------------
# Task 6: Scaling and encoding
# ------------------------------------------------------------

# HouseID is an identifier and should not be used as a predictive feature.
model_df = clean_df.drop(columns=["HouseID"])

# Separate target variable
X = model_df.drop(columns=["Price", "Log_Price"])
y = model_df["Price"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

print("\nNumeric features:", numeric_features)
print("\nCategorical features:", categorical_features)

# One-hot encoding is suitable for nominal categorical variables.
# StandardScaler is useful for algorithms sensitive to feature scale.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ],
    remainder="drop"
)

X_processed = preprocessor.fit_transform(X)

encoded_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
final_feature_names = numeric_features + list(encoded_feature_names)

processed_df = pd.DataFrame(X_processed, columns=final_feature_names, index=X.index)
processed_df["Price"] = y.values

# ------------------------------------------------------------
# Task 7: Final dataset review and save output
# ------------------------------------------------------------

print("\nFinal processed dataset shape:", processed_df.shape)
print("\nRemaining missing values:", processed_df.isna().sum().sum())
print("\nFirst 5 rows of processed dataset:")
print(processed_df.head())

processed_df.to_csv("HousePrice_Preprocessed.csv", index=False)
print("\nSaved final file as HousePrice_Preprocessed.csv")
