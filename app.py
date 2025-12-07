# app.py

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================
# 🔹 CONFIG: paths
# =========================
DATA_PATH = "retail_dataset.csv"   # <- your dataset
MODEL_PATH = "xgb_pipeline.pkl"    # <- your trained pipeline model


# =========================
# 🔹 Load data & model
# =========================
@st.cache_resource
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: str):
    p = Path(path)
    if not p.exists():
        st.error(f"❌ Model file not found: {path}")
        st.stop()
    return joblib.load(p)


df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

# =========================
# 🔹 Basic checks
# =========================
required_cols = [
    "order_id",
    "product_id",
    "customer_id",
    "category",
    "subcategory",
    "product_price",
    "discount_percent",
    "final_price",
    "order_channel",
    "payment_method",
    "delivery_days",
    "customer_tenure_days",
    "num_prior_orders",
    "num_prior_returns",
    "customer_segment",
    "region",
    "weather_at_order",
    "returned",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"These required columns are missing in your dataset: {missing}")
    st.stop()

# Features used for prediction – must match what you used in training
FEATURE_COLS = [
    "category",
    "subcategory",
    "product_price",
    "discount_percent",
    "final_price",
    "order_channel",
    "payment_method",
    "delivery_days",
    "customer_tenure_days",
    "num_prior_orders",
    "num_prior_returns",
    "customer_segment",
    "region",
    "weather_at_order",
]


# =========================
# 🔹 Streamlit UI setup
# =========================
st.set_page_config(
    page_title="Order Return Predictor",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Order Return Prediction (XGBoost Pipeline)")
st.write(
    "Use this app to **predict whether an order is likely to be returned**. "
    "You can look up an existing order from the dataset, inspect it, tweak its features, "
    "and get a prediction from your trained model."
)

# -------------------------
# Sidebar: order lookup
# -------------------------
st.sidebar.header("🔍 Order Lookup")

order_ids = df["order_id"].unique()
selected_order_id = st.sidebar.selectbox(
    "Select Order ID",
    options=sorted(order_ids),
    index=0,
)

selected_row = df[df["order_id"] == selected_order_id]

if selected_row.empty:
    st.sidebar.error("No data found for this order ID.")
    st.stop()

row = selected_row.iloc[0]  # one row as Series

st.sidebar.success(f"Loaded order **{selected_order_id}**")

# =========================
# 🔹 Show order summary
# =========================
st.subheader("📋 Order Summary from Dataset")

summary_cols = [
    "order_id",
    "product_id",
    "customer_id",
    "category",
    "subcategory",
    "product_price",
    "discount_percent",
    "final_price",
    "order_channel",
    "payment_method",
    "delivery_days",
    "customer_tenure_days",
    "num_prior_orders",
    "num_prior_returns",
    "customer_segment",
    "region",
    "weather_at_order",
    "returned",
]

st.dataframe(selected_row[summary_cols])


# =========================
# 🔹 Editable inputs
# =========================
st.subheader("🎚️ Adjust Features for Prediction")

col1, col2 = st.columns(2)

with col1:
    # Categorical text fields – default to dataset values
    category = st.text_input("Category", value=str(row["category"]))
    subcategory = st.text_input("Subcategory", value=str(row["subcategory"]))

    product_price = st.number_input(
        "Product Price",
        min_value=0.0,
        value=float(row["product_price"]),
        step=10.0,
    )

    discount_percent = st.number_input(
        "Discount (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(row["discount_percent"]),
        step=1.0,
    )

    final_price = st.number_input(
        "Final Price (after discount)",
        min_value=0.0,
        value=float(row["final_price"]),
        step=10.0,
    )

with col2:
    # Use unique values from dataset for dropdowns where possible
    order_channel_options = sorted(df["order_channel"].dropna().unique().tolist())
    payment_method_options = sorted(df["payment_method"].dropna().unique().tolist())
    customer_segment_options = sorted(df["customer_segment"].dropna().unique().tolist())
    region_options = sorted(df["region"].dropna().unique().tolist())
    weather_options = sorted(df["weather_at_order"].dropna().unique().tolist())

    def safe_index(options, value):
        try:
            return options.index(value)
        except ValueError:
            return 0 if options else 0

    order_channel = st.selectbox(
        "Order Channel",
        options=order_channel_options,
        index=safe_index(order_channel_options, row["order_channel"]),
    )

    payment_method = st.selectbox(
        "Payment Method",
        options=payment_method_options,
        index=safe_index(payment_method_options, row["payment_method"]),
    )

    delivery_days = st.number_input(
        "Delivery Days",
        min_value=0,
        max_value=60,
        value=int(row["delivery_days"]),
        step=1,
    )

    customer_tenure_days = st.number_input(
        "Customer Tenure (days)",
        min_value=0,
        max_value=3650,
        value=int(row["customer_tenure_days"]),
        step=10,
    )

    num_prior_orders = st.number_input(
        "Number of Prior Orders",
        min_value=0,
        max_value=10000,
        value=int(row["num_prior_orders"]),
        step=1,
    )

    num_prior_returns = st.number_input(
        "Number of Prior Returns",
        min_value=0,
        max_value=10000,
        value=int(row["num_prior_returns"]),
        step=1,
    )

    customer_segment = st.selectbox(
        "Customer Segment",
        options=customer_segment_options,
        index=safe_index(customer_segment_options, row["customer_segment"]),
    )

    region = st.selectbox(
        "Region",
        options=region_options,
        index=safe_index(region_options, row["region"]),
    )

    weather_at_order = st.selectbox(
        "Weather at Order Time",
        options=weather_options,
        index=safe_index(weather_options, row["weather_at_order"]),
    )

# =========================
# 🔹 Build input DataFrame
# =========================
input_data = {
    "category": category,
    "subcategory": subcategory,
    "product_price": product_price,
    "discount_percent": discount_percent,
    "final_price": final_price,
    "order_channel": order_channel,
    "payment_method": payment_method,
    "delivery_days": delivery_days,
    "customer_tenure_days": customer_tenure_days,
    "num_prior_orders": num_prior_orders,
    "num_prior_returns": num_prior_returns,
    "customer_segment": customer_segment,
    "region": region,
    "weather_at_order": weather_at_order,
}

input_df = pd.DataFrame([input_data], columns=FEATURE_COLS)

st.markdown("### 🔎 Features to be sent to the model")
st.dataframe(input_df)

# =========================
# 🔹 Predict
# =========================
if st.button("🚀 Predict Return Likelihood"):
    try:
        proba = model.predict_proba(input_df)[0]
        pred = model.predict(input_df)[0]
    except Exception as e:
        st.error(
            "❌ Error during prediction.\n\n"
            "Most common reasons:\n"
            "1. The saved model is NOT the pipeline version (it was trained on already-encoded data).\n"
            "2. The model was trained on a different set of features/columns.\n\n"
            "Make sure you trained and saved `xgb_pipeline.pkl` using the pipeline approach (preprocessor + SMOTE + model)."
        )
        st.exception(e)
    else:
        st.subheader("🧠 Prediction Result")

        classes = list(model.classes_)
        # assume '1' is the "returned" class if present, otherwise take max label
        if 1 in classes:
            pos_idx = classes.index(1)
        else:
            pos_idx = classes.index(max(classes))

        return_proba = proba[pos_idx]

        if pred == 1:
            st.error(
                f"🔴 Prediction: **ORDER WILL LIKELY BE RETURNED**\n\n"
                f"Probability of return: **{return_proba:.2f}**"
            )
        else:
            st.success(
                f"🟢 Prediction: **ORDER WILL LIKELY NOT BE RETURNED**\n\n"
                f"Probability of return: **{return_proba:.2f}**"
            )

        # Show full class probabilities
        prob_df = pd.DataFrame({
            "Class": classes,
            "Probability": proba,
        })
        st.markdown("### 📊 Class Probabilities")
        st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))

# =========================
# 🔹 Feature importance (optional, limited)
# =========================
if st.checkbox("Show Feature Importance (advanced)"):
    st.info(
        "Because the model is inside a preprocessing pipeline with one-hot encoding, "
        "feature importances are computed on many internal encoded features, not directly "
        "on the original columns. So exact mapping is approximate."
    )

    final_estimator = None
    # For imblearn Pipeline, model is usually under named_steps["model"]
    if hasattr(model, "named_steps"):
        final_estimator = model.named_steps.get("model", None)

    if final_estimator is not None and hasattr(final_estimator, "feature_importances_"):
        importances = final_estimator.feature_importances_
        st.write(f"Number of internal features after encoding: {len(importances)}")
    else:
        st.warning("Feature importances are not available for this model / pipeline.")

st.caption("Built for HCL Tech Hackathon 🧠 by Vector3.0")
