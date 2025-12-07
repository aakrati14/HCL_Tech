import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# =========================================================
# 1. Load data
# =========================================================
@st.cache_resource
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# =========================================================
# 2. Train all 4 models (LogReg, DT, RF, XGB)
# =========================================================
@st.cache_resource
def train_models(df: pd.DataFrame):
    target_col = "returned"

    # Drop target + IDs from features (if present)
    drop_cols = [target_col]
    for id_col in ["order_id", "product_id", "customer_id"]:
        if id_col in df.columns:
            drop_cols.append(id_col)

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # Split before SMOTE (important)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Identify numeric & categorical columns
    numeric_cols = [col for col in X.columns if X[col].dtype != "object"]
    categorical_cols = [col for col in X.columns if X[col].dtype == "object"]

    # Preprocessing
    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Define base models
    model_defs = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    }

    models = {}
    scores = {}

    # Train each model in its own pipeline: preprocessing -> SMOTE -> model
    for name, base_model in model_defs.items():
        pipe = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", base_model),
            ]
        )

        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)

        models[name] = pipe
        scores[name] = acc

    return models, scores, X, y, numeric_cols, categorical_cols


# =========================================================
# 3. Streamlit UI
# =========================================================
DATA_PATH = "retail_dataset.csv"

st.set_page_config(
    page_title="Order Return Prediction - Multi Model",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Order Return Prediction – Multi-Model Comparison")
st.write(
    "This app trains **4 different models** on your retail dataset and shows predictions "
    "from each of them for the same order details."
)

df = load_data(DATA_PATH)

required_cols = [
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

models, scores, X_full, y_full, numeric_cols, categorical_cols = train_models(df)

st.sidebar.header("🔍 Order Lookup")

# Use order_id if available, else let user pick row index
if "order_id" in df.columns:
    order_ids = df["order_id"].unique()
    selected_order_id = st.sidebar.selectbox(
        "Select Order ID", options=sorted(order_ids)
    )
    selected_row = df[df["order_id"] == selected_order_id]
    row = selected_row.iloc[0]
else:
    idx = st.sidebar.number_input(
        "Select Row Index", min_value=0, max_value=len(df) - 1, value=0, step=1
    )
    selected_row = df.iloc[[idx]]
    row = selected_row.iloc[0]
    selected_order_id = f"index {idx}"

st.sidebar.success(f"Loaded order: {selected_order_id}")

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
summary_cols = [c for c in summary_cols if c in df.columns]
st.dataframe(selected_row[summary_cols])

# =========================================================
# 4. Editable inputs for prediction
# =========================================================
st.subheader("🎚️ Adjust Features for Prediction")

col1, col2 = st.columns(2)

with col1:
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
        max_value=99.99,
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

# Build input DataFrame
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

input_df = pd.DataFrame([input_data])

st.markdown("### 🔎 Features sent to all models")
st.dataframe(input_df)

# =========================================================
# 5. Predict with all models
# =========================================================
if st.button("🚀 Predict with All Models"):
    results = []
    for model_name, model in models.items():
        try:
            proba = model.predict_proba(input_df)[0]
            pred = model.predict(input_df)[0]

            classes = list(model.classes_)
            # assume '1' is positive class if present, else max label
            if 1 in classes:
                pos_idx = classes.index(1)
            else:
                pos_idx = classes.index(max(classes))

            return_proba = proba[pos_idx]
            results.append({
                "Model": model_name,
                "Prediction": "Returned" if pred == 1 else "Not Returned",
                "Return_Probability": return_proba,
                "Test_Accuracy": scores[model_name],
            })
        except Exception as e:
            results.append({
                "Model": model_name,
                "Prediction": "ERROR",
                "Return_Probability": np.nan,
                "Test_Accuracy": scores[model_name],
            })
            st.error(f"Error while predicting with {model_name}: {e}")

    if results:
        res_df = pd.DataFrame(results)
        res_df["Return_Probability"] = res_df["Return_Probability"].map(
            lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
        )
        res_df["Test_Accuracy"] = res_df["Test_Accuracy"].map(lambda x: f"{x:.2f}")

        st.subheader("🧠 Model Comparison")
        st.dataframe(res_df)

        st.caption("Return_Probability = model's predicted probability that the order will be returned.")
