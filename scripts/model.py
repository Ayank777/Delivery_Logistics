# ==========================================
# Machine Learning - Final Tuned Version
# Random Forest + XGBoost + Full Evaluation
# ==========================================

import pandas as pd
import random

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("output/cleaned_data.csv")

print("✅ Data Loaded\n")

# Keep original (for display)
df_original = df.copy()

# -------------------------------
# Encode categorical columns
# -------------------------------
df_encoded = df.copy()

le_dict = {}

for col in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    le_dict[col] = le

print("🔤 Encoding completed\n")

# -------------------------------
# Features & Target
# -------------------------------
X = df_encoded.drop(
    ['Delayed', 'delayed', 'delivery_status', 'delivery_id'],
    axis=1,
    errors='ignore'
)

y = df_encoded['Delayed']

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("📊 Training size:", X_train.shape)
print("📊 Testing size:", X_test.shape)

# ==========================================
# 🌳 TUNED RANDOM FOREST MODEL
# ==========================================
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

# Training accuracy
rf_train_pred = rf_model.predict(X_train)
rf_train_acc = accuracy_score(y_train, rf_train_pred)

# Testing accuracy
rf_test_acc = accuracy_score(y_test, rf_pred)

print("\n🌳 RANDOM FOREST RESULTS (TUNED)")
print("Training Accuracy:", rf_train_acc)
print("Testing Accuracy :", rf_test_acc)

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# ==========================================
# 🚀 TUNED XGBOOST MODEL
# ==========================================
xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

# Training accuracy
xgb_train_pred = xgb_model.predict(X_train)
xgb_train_acc = accuracy_score(y_train, xgb_train_pred)

# Testing accuracy
xgb_test_acc = accuracy_score(y_test, xgb_pred)

print("\n🚀 XGBOOST RESULTS (TUNED)")
print("Training Accuracy:", xgb_train_acc)
print("Testing Accuracy :", xgb_test_acc)

print("\nClassification Report:")
print(classification_report(y_test, xgb_pred))

# ==========================================
# 🔮 DEMO 1: RANDOM REAL-WORLD PREDICTION
# ==========================================
print("\n🔮 DEMO 1: RANDOM PREDICTION")

idx = random.choice(X_test.index)

sample_original = df_original.loc[idx]

clean_display = sample_original.drop(
    ['Delayed', 'delayed', 'delivery_status', 'delivery_id'],
    errors='ignore'
)

print("\n📥 Input (Readable & Clean):")
print(clean_display)

sample_encoded = df_encoded.loc[idx].drop(
    ['Delayed', 'delayed', 'delivery_status', 'delivery_id'],
    errors='ignore'
)

prediction = rf_model.predict(pd.DataFrame([sample_encoded]))[0]

print("\nPrediction:", prediction)

if prediction == 1:
    print("🚨 Delivery will be DELAYED")
else:
    print("✅ Delivery will be ON TIME")

# ==========================================
# 🔮 DEMO 2: ON-TIME CASE (VALIDATION)
# ==========================================
print("\n🔮 DEMO 2: ON-TIME CASE")

ontime_index = df_encoded[df_encoded['Delayed'] == 0].index[0]

sample_original_ontime = df_original.loc[ontime_index]

clean_display_ontime = sample_original_ontime.drop(
    ['Delayed', 'delayed', 'delivery_status', 'delivery_id'],
    errors='ignore'
)

print("\n📥 Input (Readable & Clean):")
print(clean_display_ontime)

sample_encoded_ontime = df_encoded.loc[ontime_index].drop(
    ['Delayed', 'delayed', 'delivery_status', 'delivery_id'],
    errors='ignore'
)

prediction_ontime = rf_model.predict(pd.DataFrame([sample_encoded_ontime]))[0]

print("\nPrediction:", prediction_ontime)

if prediction_ontime == 1:
    print("🚨 Delivery will be DELAYED")
else:
    print("✅ Delivery will be ON TIME")

# ==========================================
# 🔮 DEMO 3: DELAYED CASE (VALIDATION)
# ==========================================
print("\n🔮 DEMO 3: DELAYED CASE")

delayed_index = df_encoded[df_encoded['Delayed'] == 1].index[0]

sample_original_delayed = df_original.loc[delayed_index]

clean_display_delayed = sample_original_delayed.drop(
    ['Delayed', 'delayed', 'delivery_status', 'delivery_id'],
    errors='ignore'
)

print("\n📥 Input (Readable & Clean):")
print(clean_display_delayed)

sample_encoded_delayed = df_encoded.loc[delayed_index].drop(
    ['Delayed', 'delayed', 'delivery_status', 'delivery_id'],
    errors='ignore'
)

prediction_delayed = rf_model.predict(pd.DataFrame([sample_encoded_delayed]))[0]

print("\nPrediction:", prediction_delayed)

if prediction_delayed == 1:
    print("🚨 Delivery will be DELAYED")
else:
    print("✅ Delivery will be ON TIME")