import streamlit as st
import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix, classification_report

# Load the trained model
model_path = "stacking_model.pkl"  # Pastikan jalur model sudah benar
model = load(model_path)

# Fungsi untuk memproses data dan melakukan prediksi
def process_and_predict(data, model):
    # Periksa dan bersihkan kolom 'is_fraud' untuk memastikan nilainya valid (0 atau 1)
    data['is_fraud'] = pd.to_numeric(data['is_fraud'], errors='coerce')  # Mengubah ke numerik, nilai yang salah menjadi NaN
    data = data.dropna(subset=['is_fraud'])  # Menghapus baris yang memiliki NaN di kolom 'is_fraud'

    # Siapkan X dan y dengan menghapus kolom yang tidak diperlukan dan mengkodekan fitur kategorikal
    X = data.drop(columns=["is_fraud", "trans_date_trans_time", "trans_num", "dob"], errors='ignore')
    y = data["is_fraud"].astype(int)
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Tambahkan kolom yang hilang sesuai dengan fitur yang diharapkan oleh model
    for col in model.feature_names_in_:
        if col not in X_encoded.columns:
            X_encoded[col] = 0  # Tambahkan kolom yang hilang dengan nilai default 0

    # Sesuaikan urutan kolom agar sesuai dengan yang digunakan pada pelatihan model
    X_encoded = X_encoded[model.feature_names_in_]

    # Prediksi menggunakan model
    y_pred = model.predict(X_encoded)

    # Hitung metrik evaluasi
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    report = classification_report(y, y_pred, output_dict=True)

    # Kumpulkan hasil untuk ditampilkan
    results = {
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "classification_report": pd.DataFrame(report).T,
        "sample_predictions": data.assign(predicted_fraud=y_pred).head(),
        "prediction_summary": {
            "total_fraud": int(sum(y_pred)),
            "total_non_fraud": int(len(y_pred) - sum(y_pred))
        },
        # Pisahkan data yang diprediksi sebagai fraud dan non-fraud
        "fraud_data": data.assign(predicted_fraud=y_pred).loc[y_pred == 1],
        "non_fraud_data": data.assign(predicted_fraud=y_pred).loc[y_pred == 0]
    }
    
    return results

# Muat data dan tampilkan hasil jika file diunggah
uploaded_file = st.file_uploader("Upload data Uji pada dashboard (CSV format)", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.ExcelFile(uploaded_file).parse(0)

    # Proses data dan lakukan prediksi
    results = process_and_predict(data, model)

    # Tampilkan hasil evaluasi model
    st.title("Fraud Detection Model Evaluation")

    # Tampilkan confusion matrix
    st.write("### Confusion Matrix")
    st.write(f"True Positives (Prediksi benar sebagai penipuan.): {results['confusion_matrix']['tp']}")
    st.write(f"False Positives (Prediksi sebagai penipuan, tetapi sebenarnya bukan.): {results['confusion_matrix']['fp']}")
    st.write(f"True Negatives (Prediksi benar sebagai bukan penipuan): {results['confusion_matrix']['tn']}")
    st.write(f"False Negatives (Prediksi sebagai bukan penipuan, tetapi sebenarnya penipuan.): {results['confusion_matrix']['fn']}")

    # Tampilkan classification report
    st.write("### Classification Report")
    st.dataframe(results["classification_report"])

    # Tampilkan prediksi sampel
    st.write("### Sample Predictions")
    st.dataframe(results["sample_predictions"])

    # Tampilkan ringkasan prediksi
    st.write("### Summary of Predictions")
    st.write(f"Total transactions predicted as fraud: {results['prediction_summary']['total_fraud']}")
    st.write(f"Total transactions predicted as non-fraud: {results['prediction_summary']['total_non_fraud']}")

    # Tampilkan tabel data yang terdeteksi fraud dan tidak fraud
    st.write("### Detected Fraud Transactions")
    st.dataframe(results["fraud_data"])

    st.write("### Detected Non-Fraud Transactions")
    st.dataframe(results["non_fraud_data"])
