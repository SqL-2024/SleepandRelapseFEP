import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import joblib
# ======================
# Load trained model
# ======================
MODEL_PATH = "deployed_rf_model_sleep_5assessments.pkl"   # ← 改成你的 pkl 路径
THRESHOLD = 0.5                  # ← 如果你有 Youden 阈值，在这里替换


model = joblib.load(MODEL_PATH)

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Relapse Prediction (within 12 months)", layout="centered")

st.title("🧠 Relapse Risk Prediction (within 12 months) in First Episode Psychosis")
st.write(
    """
    This tool predicts the risk of relapse within 12 months based on sleep characteristics assessed in the first 30 days.

    Note: For more reliable results, we recommend completing as much of the 30-day sleep data collection as possible and carefully checking that all entered information is accurate.
    This tool is a research prototype and is not intended for clinical use.
    Please interpret individual predictions with care and consider discussing the results with a healthcare professional.
    """
)



st.markdown("---")

# ======================
# User input
# ======================
st.subheader("📋 Input Features")

def user_input_features():
    data = {}
    data["sleep_coverage"] = st.slider(
        "Sleep coverage",
        0.0, 1.0, 0.8,
        help="Proportion of days with available sleep data during the assessment period."
    )

    data["trouble_falling_asleep_rate"] = st.slider(
        "Trouble falling asleep rate",
        0.0, 1.0, 0.1,
        help="Proportion of days on which difficulty falling asleep was reported."
    )

    data["wakeup_frequently_rate"] = st.slider(
        "Wake up frequently rate",
        0.0, 1.0, 0.1,
        help="Proportion of days on which frequent awakenings during sleep were reported."
    )

    data["insufficient_sleep_rate"] = st.slider(
        "Insufficient sleep rate",
        0.0, 1.0, 0.1,
        help="Proportion of days on which not have enough sleep  were reported."
    )

    data["sleep_hours_mean"] = st.number_input(
        "Mean sleep hours",
        0.0, 24.0, 7.0,
        help="Average number of hours slept per night across the assessment period."
    )

    data["sleep_hours_std"] = st.number_input(
        "Sleep hours STD",
        0.0, 10.0, 1.0,
        help="Variability (standard deviation) of nightly sleep duration across the assessment period."
    )

    data["short_sleep_rate"] = st.slider(
        "Short sleep rate",
        0.0, 1.0, 0.1,
        help="Proportion of days with sleep duration less than 7 hours."
    )

    data["sleep_problem_rate"] = st.slider(
        "Sleep problem rate",
        0.0, 1.0, 0.1,
        help="Proportion of days on which at least one sleep problem was reported."
    )

    data["sleep_problem_burden"] = st.number_input(
        "Sleep problem burden",
        0.0, 3.0, 1.0,
        help="Average number of reported sleep problems per day, with multiple problems on the same day counted separately."
    )


    return pd.DataFrame([data])


input_df = user_input_features()

st.markdown("---")
st.subheader("🔎 Input Summary")
st.dataframe(input_df)
st.markdown("---")
st.subheader("📈 Global Feature Importance (SHAP)")

st.image(
    "deployed_model_shap_summary.png",
    caption="Mean absolute SHAP values from the training data of the deployed model.",
    use_column_width=True
)

# ======================
# Prediction
# ======================
if st.button("🚀 Predict Relapse Risk"):
    prob = model.predict_proba(input_df)[:, 1][0]
    pred = int(prob >= THRESHOLD)

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    st.write(f"**Predicted probability of relapse:** `{prob:.3f}`")

    if pred == 1:
        st.error("⚠️ High risk of relapse within 12 months")
    else:
        st.success("✅ Low risk of relapse within 12 months")

    st.caption(f"Decision threshold = {THRESHOLD}")

    # ======================
    # SHAP explanation
    # ======================
    st.markdown("---")
    st.subheader("🔍 Explanation for This Prediction (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # 只解释正类（relapse）
    shap_val_single = shap_values[0, :, 1]  # 第一个样本、第二个类别（relapse）

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_val_single,
            base_values=explainer.expected_value[1],
            data=input_df.iloc[0],
            feature_names=input_df.columns
        ),
        max_display=10
    )

    st.pyplot(fig)

    # st.subheader("⚡ SHAP Force Plot for This Prediction")
    # force_plot_html = shap.force_plot(
    #     explainer.expected_value[1],
    #     shap_values[0, :, 1],
    #     input_df.iloc[0],
    #     feature_names=input_df.columns,
    #     matplotlib=False
    # ).data  # 获取 HTML

    # st.components.v1.html(force_plot_html, height=300)


