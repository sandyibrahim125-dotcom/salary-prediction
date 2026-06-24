"""
Streamlit entry point for the Salary Prediction app.

Run locally with:
    streamlit run app.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import streamlit as st

from model.predict import inference, get_feature_metadata

st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="centered",
)

st.title("💼 Employee Salary Predictor")
st.markdown(
    """
    Estimate an employee's expected annual salary based on **years of
    experience**, **education level**, **job title**, **age**, **gender**,
    and **location**. The model was trained on a salary dataset using a
    tuned regression pipeline (see the accompanying Jupyter notebook for
    full details on training and evaluation).
    """
)

metadata = get_feature_metadata()

st.divider()
st.subheader("Enter Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=30,
        step=1,
        help="Employee's age in years.",
    )
    years_experience = st.number_input(
        "Years of Experience",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
        help="Total years of professional experience.",
    )
    gender = st.selectbox("Gender", options=metadata["categorical_options"]["Gender"])

with col2:
    education_level = st.selectbox(
        "Education Level",
        options=metadata["categorical_options"]["Education Level"],
    )
    job_title = st.selectbox(
        "Job Title",
        options=sorted(metadata["categorical_options"]["Job Title"]),
    )
    location = st.selectbox(
        "Location",
        options=metadata["categorical_options"]["Location"],
    )

st.divider()

if st.button("Predict Salary", type="primary", use_container_width=True):
    raw_input = {
        "Age": age,
        "Years of Experience": years_experience,
        "Gender": gender,
        "Education Level": education_level,
        "Job Title": job_title,
        "Location": location,
    }

    result = inference(raw_input)

    for warning in result["warnings"]:
        st.warning(warning)

    st.success("Prediction complete!")

    predicted = result["predicted_salary"]
    low = result["confidence_low"]
    high = result["confidence_high"]

    st.metric(label="Estimated Annual Salary", value=f"${predicted:,.0f}")

    if low is not None and high is not None:
        st.caption(f"Estimated range (80% of model trees): **${low:,.0f} – ${high:,.0f}**")

        chart_df = pd.DataFrame({
            "Scenario": ["Lower estimate", "Predicted", "Upper estimate"],
            "Salary": [low, predicted, high],
        })
        st.bar_chart(chart_df.set_index("Scenario"))

    with st.expander("Model performance on held-out test data"):
        st.write(f"**MAE:** ${metadata['test_mae']:,.2f}")
        st.write(f"**RMSE:** ${metadata['test_rmse']:,.2f}")
        st.write(f"**R² score:** {metadata['test_r2']:.4f}")
        st.caption(
            "These metrics describe how accurate the model is on data it "
            "was not trained on, computed in the training notebook."
        )

st.divider()
st.caption(
    "Built with Streamlit · Model: scikit-learn regression pipeline · "
    "See README.md for dataset details and how to run this app locally."
)
