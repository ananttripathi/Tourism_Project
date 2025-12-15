# Tourism Package Prediction - MLOps Pipeline Assignment

## 📋 Project Overview

This project implements an end-to-end MLOps pipeline for predicting whether customers will purchase the **Wellness Tourism Package** from "Visit with Us" travel company.

## 🗂️ Project Structure

```
tourism_project/
├── .github/
│   └── workflows/
│       └── pipeline.yml                 # GitHub Actions CI/CD workflow
├── data/
│   └── tourism.csv                      # Original dataset
├── deployment/
│   ├── app.py                           # Streamlit web application
│   ├── Dockerfile                       # Docker configuration
│   └── requirements.txt                 # Deployment dependencies
├── hosting/
│   └── hosting.py                       # Script to push to Hugging Face Spaces
├── model_building/
│   ├── data_register.py                 # Dataset registration to Hugging Face
│   ├── prep.py                          # Data preprocessing script
│   └── train.py                         # Model training with MLflow tracking
└── requirements.txt                     # Workflow dependencies
```

## 🎯 Key Features

### 1. **Data Registration & Preparation**
- Automated dataset upload to Hugging Face Hub
- Comprehensive data cleaning and preprocessing
- Handling of missing values and data quality issues
- Label encoding of categorical variables
- Stratified train-test split (80-20)

### 2. **Model Training**
- **Algorithm**: XGBoost Classifier
- **Hyperparameter Tuning**: GridSearchCV with 3-fold cross-validation
- **Experiment Tracking**: MLflow integration
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 3. **Deployment**
- **Web Application**: Interactive Streamlit app
- **Containerization**: Docker support
- **Hosting**: Hugging Face Spaces

### 4. **CI/CD Pipeline**
- Automated workflow with GitHub Actions
- Four main jobs:
  1. Dataset Registration
  2. Data Preparation
  3. Model Training
  4. Deployment to Hugging Face

## 🚀 Getting Started

### Prerequisites

Before running this project, you need to set up the following:

#### 1. **GitHub Repository Setup**
```bash
# Create a new GitHub repository
# Repository name: tourism-package-prediction (or your choice)
# Initialize with README
```

#### 2. **Hugging Face Setup**
- Create a Hugging Face account at https://huggingface.co
- Generate an access token:
  - Go to Settings → Access Tokens
  - Create a new token with **Write** permissions
  - Copy and save the token securely

#### 3. **GitHub Secrets Configuration**
- Go to your GitHub repository
- Navigate to: Settings → Secrets and Variables → Actions
- Add a new repository secret:
  - **Name**: `HF_TOKEN`
  - **Secret**: Paste your Hugging Face token

#### 4. **Hugging Face Spaces**
Create the following spaces on Hugging Face:
- **Space name**: `wellness-tourism-prediction`
- **SDK**: Docker (Streamlit template)

### 📝 Before Pushing to GitHub

**IMPORTANT**: You must replace all placeholder values in the code:

Replace `<---repo id---->` with your Hugging Face username in the following files:

1. **tourism_project/model_building/data_register.py**
   - Line: `repo_id = "<---repo id---->/tourism-dataset"`

2. **tourism_project/model_building/prep.py**
   - Line: `DATASET_PATH = "hf://datasets/<---repo id---->/tourism-dataset/tourism.csv"`
   - Line: `repo_id="<---repo id---->/tourism-dataset"`

3. **tourism_project/model_building/train.py**
   - Lines with paths to Hugging Face datasets
   - Line: `repo_id = "<---repo id---->/tourism-prediction-model"`

4. **tourism_project/deployment/app.py**
   - Line: `repo_id="<---repo id---->/tourism-prediction-model"`

5. **tourism_project/hosting/hosting.py**
   - Line: `repo_id="<---repo id---->/wellness-tourism-prediction"`

### 🔧 Installation

```bash
# Clone your GitHub repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Install dependencies
pip install -r tourism_project/requirements.txt
```

### 🏃 Running Locally

#### Option 1: Run Individual Scripts

```bash
# 1. Register dataset (requires HF_TOKEN environment variable)
export HF_TOKEN="your_hugging_face_token"  # On Windows: set HF_TOKEN=your_token
python tourism_project/model_building/data_register.py

# 2. Prepare data
python tourism_project/model_building/prep.py

# 3. Train model (start MLflow server first)
mlflow ui --host 0.0.0.0 --port 5000 &
python tourism_project/model_building/train.py

# 4. Deploy to Hugging Face
python tourism_project/hosting/hosting.py
```

#### Option 2: Run Streamlit App Locally

```bash
cd tourism_project/deployment
streamlit run app.py
```

### 📤 Deploying via GitHub Actions

1. **Prepare your repository**:
   - Ensure all placeholder values are replaced
   - Verify HF_TOKEN is added to GitHub Secrets

2. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Initial commit: Tourism Package Prediction Pipeline"
   git push origin main
   ```

3. **Monitor the workflow**:
   - Go to your GitHub repository
   - Click on "Actions" tab
   - Watch the pipeline execute automatically

4. **Access your deployed app**:
   - Visit: `https://huggingface.co/spaces/<your-username>/wellness-tourism-prediction`

## 📊 Model Performance

The model is trained using XGBoost Classifier with the following characteristics:

- **Training Features**: 17 features including customer demographics and interaction data
- **Target Variable**: Binary (Purchase = 1, No Purchase = 0)
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC Score

## 🎨 Streamlit Application Features

The web application provides:
- **User-friendly interface** with two-column layout
- **Customer Demographics Section**: Age, occupation, income, etc.
- **Interaction Data Section**: Pitch details, follow-ups, preferences
- **Real-time Predictions** with confidence scores
- **Actionable Recommendations** based on prediction results

## 🔬 MLflow Experiment Tracking

All experiments are tracked using MLflow:
- Hyperparameter combinations logged
- Model metrics recorded
- Best model artifacts saved
- Experiment comparison capabilities

## 📦 Dataset Description

**Features**:
- Customer demographics (Age, Gender, Occupation, etc.)
- Travel preferences (PropertyStar, NumberOfTrips, etc.)
- Interaction data (PitchDuration, Followups, SatisfactionScore)
- Financial data (MonthlyIncome, OwnCar)

**Target**: `ProdTaken` (0 = No Purchase, 1 = Purchase)

## 🛠️ Technologies Used

- **Python 3.9**
- **XGBoost** - Machine Learning
- **scikit-learn** - Preprocessing & Evaluation
- **MLflow** - Experiment Tracking
- **Streamlit** - Web Application
- **Docker** - Containerization
- **GitHub Actions** - CI/CD
- **Hugging Face Hub** - Model & Data Storage

## 📝 Assignment Submission Checklist

- [x] Complete folder structure created
- [x] Data registration script (`data_register.py`)
- [x] Data preparation script (`prep.py`)
- [x] Model training script with MLflow (`train.py`)
- [x] Streamlit application (`app.py`)
- [x] Dockerfile for deployment
- [x] Deployment requirements.txt
- [x] Hosting script (`hosting.py`)
- [x] GitHub Actions workflow (`pipeline.yml`)
- [x] Workflow requirements.txt
- [x] Jupyter notebook with all code cells filled
- [ ] Replace all `<---repo id---->` placeholders with your HF username
- [ ] GitHub repository created
- [ ] HF_TOKEN added to GitHub Secrets
- [ ] Hugging Face Space created
- [ ] Pipeline executed successfully
- [ ] Screenshots of:
  - GitHub repository structure
  - GitHub Actions workflow execution
  - Deployed Streamlit app on Hugging Face

## 📸 Output Requirements

### 1. GitHub Repository
- Screenshot showing folder structure
- Screenshot showing successful workflow execution

### 2. Hugging Face Space
- Link to deployed application
- Screenshot of the Streamlit app in action

## 🔍 Troubleshooting

### Common Issues:

1. **HF_TOKEN not found**:
   - Ensure the token is added to GitHub Secrets
   - Verify the secret name is exactly `HF_TOKEN`

2. **Import errors**:
   - Check all dependencies are in requirements.txt
   - Verify correct versions are specified

3. **Model not loading in Streamlit**:
   - Ensure model is uploaded to correct HF repository
   - Check repo_id matches across files

4. **GitHub Actions failing**:
   - Check workflow logs for specific errors
   - Verify all file paths are correct
   - Ensure requirements.txt includes all dependencies

## 👨‍💻 Author

This MLOps pipeline was developed as part of the Advanced Machine Learning and MLOps course assignment.

## 📄 License

This project is for educational purposes as part of the course assignment.

---

**Note**: Remember to replace ALL placeholder values (`<---repo id---->`) with your actual Hugging Face username before submitting!

