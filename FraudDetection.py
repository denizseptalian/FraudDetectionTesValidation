import streamlit as st
import pandas as pd
import joblib
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import sklearn

# Cek versi scikit-learn
st.write(f"Using scikit-learn version: {sklearn.__version__}")

# Load the trained model dengan penanganan error
model_path = "stacking_model.pkl"
try:
    model = joblib.load(model_path)
except AttributeError:
    st.warning("Joblib loading error detected. Trying pickle load...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Fungsi untuk memproses data dan melakukan prediksi
def process_and_predict(data, model):
    try:
        data['is_fraud'] = pd.to_numeric(data['is_fraud'], errors='coerce')
        data = data.dropna(subset=['is_fraud'])
        X = data.drop(columns=["is_fraud", "trans_date_trans_time", "trans_num", "dob"], errors='ignore')
        y = data["is_fraud"].astype(int)
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Pastikan fitur sesuai dengan model
        for col in model.feature_names_in_:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[model.feature_names_in_]
        
        # Prediksi
        y_pred = model.predict(X_encoded)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        report = classification_report(y, y_pred, output_dict=True)

        results = {
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "classification_report": pd.DataFrame(report).T,
            "sample_predictions": data.assign(predicted_fraud=y_pred).head(),
            "prediction_summary": {
                "total_fraud": int(sum(y_pred)),
                "total_non_fraud": int(len(y_pred) - sum(y_pred))
            },
            "fraud_data": data.assign(predicted_fraud=y_pred).loc[y_pred == 1],
            "non_fraud_data": data.assign(predicted_fraud=y_pred).loc[y_pred == 0]
        }
        return results
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None

# Muat data jika file diunggah
uploaded_file = st.file_uploader("Upload data Uji (CSV/XLSX)", type=["csv", "xlsx"])
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        results = process_and_predict(data, model)
        if results:
            st.title("Fraud Detection Model Evaluation")
            st.write("### Confusion Matrix")
            st.write(results['confusion_matrix'])
            st.write("### Classification Report")
            st.dataframe(results["classification_report"])
            st.write("### Sample Predictions")
            st.dataframe(results["sample_predictions"])
            st.write("### Summary of Predictions")
            st.write(results["prediction_summary"])
            st.write("### Detected Fraud Transactions")
            st.dataframe(results["fraud_data"])
            st.write("### Detected Non-Fraud Transactions")
            st.dataframe(results["non_fraud_data"])
    except Exception as e:
        st.error(f"Error loading file: {e}")
