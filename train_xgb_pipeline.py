import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# 1. Load data
df = pd.read_csv("retail_dataset.csv")

target_col = "returned"

X = df.drop(columns=[target_col, "order_id", "product_id", "customer_id"])
y = df[target_col]

# 2. Identify column types
numeric_cols = [
    "product_price",
    "discount_percent",
    "final_price",
    "delivery_days",
    "customer_tenure_days",
    "num_prior_orders",
    "num_prior_returns",
]

categorical_cols = [
    "category",
    "subcategory",
    "order_channel",
    "payment_method",
    "customer_segment",
    "region",
    "weather_at_order",
]

# 3. Preprocessing
numeric_transformer = SimpleImputer(strategy="median")

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# 4. XGBoost Model
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

# 5. FULL Pipeline — preprocessing → SMOTE → XGBoost
clf = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", xgb),
    ]
)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Train the pipeline
clf.fit(X_train, y_train)

print("Training accuracy:", clf.score(X_train, y_train))
print("Testing accuracy:", clf.score(X_test, y_test))

# 8. Save the full pipeline
joblib.dump(clf, "xgb_pipeline.pkl")
print("Saved pipeline as xgb_pipeline.pkl")
