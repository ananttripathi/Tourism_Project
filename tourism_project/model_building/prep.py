# for data manipulation
import pandas as pd
import numpy as np
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
import os

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/<---repo id---->/tourism-dataset/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

# Drop the unnamed index column if it exists
if 'Unnamed: 0' in df.columns or df.columns[0] == '':
    df = df.iloc[:, 1:]

# Drop CustomerID as it's a unique identifier (not useful for modeling)
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# Handle missing values
print("\nHandling missing values...")
# For numerical columns, fill with median
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# For categorical columns, fill with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Handle specific data quality issues (e.g., "Fe Male" should be "Female")
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].str.strip().replace({'Fe Male': 'Female', 'Fe male': 'Female'})

# Encode categorical columns
print("\nEncoding categorical variables...")
label_encoder = LabelEncoder()

# List of categorical columns to encode
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 
                        'MaritalStatus', 'Designation']

for col in categorical_features:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {Xtrain.shape[0]}")
print(f"Test set size: {Xtest.shape[0]}")

# Save the datasets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("\nDatasets saved locally.")

# Upload to Hugging Face
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="<---repo id---->/tourism-dataset",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to Hugging Face")

print("\nData preparation completed successfully!")

