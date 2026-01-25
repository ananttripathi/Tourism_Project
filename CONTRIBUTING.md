# Contributing to Tourism Package Prediction

Thanks for your interest in this project. This MLOps pipeline was built for the **Advanced Machine Learning and MLOps** course and is shared for portfolio and learning.

## Quick links

- **Live demo:** [Hugging Face Spaces](https://huggingface.co/spaces/ananttripathiak/wellness-tourism-prediction)
- **Setup guide:** [README – Getting Started](README.md#-getting-started)

## How to run locally

1. **Clone and install**
   ```bash
   git clone https://github.com/ananttripathi/Tourism_Project.git
   cd Tourism_Project
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r tourism_project/requirements.txt
   ```

2. **Pipeline steps** (see [README – Running Locally](README.md#-running-locally)):
   - Set `HF_TOKEN` and run `data_register.py` → `prep.py` → `train.py`
   - Or use GitHub Actions (add `HF_TOKEN` to repo secrets) and push to `main`.

3. **Run Streamlit app**
   ```bash
   cd tourism_project/deployment && streamlit run app.py
   ```

## CI/CD

The pipeline runs on push to `main`. Ensure `HF_TOKEN` is in **Settings → Secrets and variables → Actions**.

## Feedback

Open a [GitHub Issue](https://github.com/ananttripathi/Tourism_Project/issues) for bugs or ideas.
