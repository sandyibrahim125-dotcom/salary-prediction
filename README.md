# 💼 Salary Prediction

A machine learning project that predicts an employee's annual salary based
on years of experience, education level, job title, age, gender, and
location. Includes a full experimentation notebook and a Streamlit web app
for interactive predictions.

## Project Description

This project trains a regression pipeline (preprocessing + model bundled
together with scikit-learn) to estimate salary from employee attributes.
The full experimentation process — exploratory data analysis, preprocessing,
baseline modeling, hyperparameter tuning with cross-validation, and
evaluation — is documented in `notebook/salary_prediction.ipynb`. The final
trained pipeline is exported and served through a Streamlit app
(`app.py`) where users can enter employee details and get an instant salary
estimate with an approximate confidence range.

**Final model performance (held-out test set):**
- MAE: ~$7,481
- RMSE: ~$9,525
- R²: ~0.905

## Demo

> Replace this with your deployed Streamlit Cloud URL after deployment:
> `https://<your-app-name>.streamlit.app`

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/salary-prediction.git
   cd salary-prediction
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   The app will open automatically at `http://localhost:8501`.

5. (Optional) Re-run the training notebook:
   ```bash
   jupyter notebook notebook/salary_prediction.ipynb
   ```
   This regenerates `model/salary_model.pkl` and `model/metadata.json` from
   scratch.

## Project Structure

```
project_root/
├── app.py                          # Streamlit entry point
├── model/
│   ├── __init__.py
│   ├── predict.py                  # Loads model, implements inference(data)
│   ├── preprocess.py                # Input validation & DataFrame shaping
│   ├── salary_model.pkl             # Exported trained pipeline
│   └── metadata.json                # Feature metadata for the UI
├── notebook/
│   └── salary_prediction.ipynb      # Full EDA -> training -> evaluation pipeline
├── data/
│   └── salary_data.csv              # Training dataset
├── requirements.txt
├── README.md
├── CONTRIBUTIONS.md
└── .gitignore
```

## Used Dataset

The dataset schema mirrors the public Kaggle dataset **"Salary Prediction
Dataset"** (age, gender, education level, job title, years of experience,
salary) — see
https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer.
For this project, a `Location` column was added to enrich the feature set
and a synthetic dataset (`data/salary_data.csv`, 1,200 rows) was generated
following the same structure and realistic feature-salary relationships, so
the project is fully reproducible without requiring Kaggle credentials.

Columns:

| Column | Type | Description |
|---|---|---|
| Age | numeric | Employee age in years |
| Gender | categorical | Male / Female |
| Education Level | categorical | High School / Bachelor's / Master's / PhD |
| Job Title | categorical | One of 20 common job titles |
| Years of Experience | numeric | Total years of professional experience |
| Location | categorical | One of 10 cities / "Remote" |
| Salary | numeric (target) | Annual salary in USD |

If you'd like to use the original Kaggle CSV instead, download it and place
it at `data/salary_data.csv` with matching column names, then re-run the
notebook.

## Tech Stack

- **Modeling:** scikit-learn (RandomForestRegressor / GradientBoostingRegressor, GridSearchCV)
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **App:** Streamlit
- **Persistence:** joblib

## License

This project is for educational purposes.
