import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import random
import os
import smtplib
from email.message import EmailMessage
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load trained model
model_bundle = joblib.load("Models/ids_model.pkl")
model = model_bundle["model"]  # this is your classifier
scaler = model_bundle["scaler"]  # if you're using it


# Email Alert Setup (Update your credentials)
EMAIL_ADDRESS = "athangsinhale2004@gmail.com"
EMAIL_PASSWORD = "elszrhnabkmdifqc"
TO_EMAIL = "absinhale2004@gmail.com"

# ---------------- Helper Functions ---------------- #

def preprocess_input(data):
    # Drop label if present
    if 'Label' in data.columns:
        data = data.drop('Label', axis=1)

    # Drop all non-numeric columns
    non_numeric_cols = data.select_dtypes(include=['object']).columns
    data = data.drop(columns=non_numeric_cols, errors='ignore')

    # Convert to numeric, coerce bad data
    data = data.apply(pd.to_numeric, errors='coerce')
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    # Get expected features from model bundle
    expected_features = model_bundle["feature_names"]

    # Add missing columns with zeros
    for col in expected_features:
        if col not in data.columns:
            data[col] = 0

    # Ensure correct column order
    data = data[expected_features]

    # Scale the data
    return scaler.transform(data)


def generate_fake_traffic():
    return pd.DataFrame([{
        'duration': np.random.randint(1, 100),
        'protocol_type': random.choice([0, 1, 2]),
        'service': random.choice([0, 1, 2, 3]),
        'flag': random.choice([0, 1, 2]),
        'src_bytes': np.random.randint(0, 10000),
        'dst_bytes': np.random.randint(0, 10000),
        'wrong_fragment': np.random.randint(0, 3),
        'hot': np.random.randint(0, 10)
    }])

def send_email_alert(threat_info):
    msg = EmailMessage()
    msg["Subject"] = "🚨 Intrusion Detected!"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL
    msg.set_content(f"An intrusion has been detected:\n\n{threat_info}")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

# Log file path
log_file = "intrusion_logs.csv"
if not os.path.exists(log_file):
    pd.DataFrame(columns=["Time", "Duration", "Protocol", "Service", "Flag", "Src_Bytes", "Dst_Bytes", "Threat_Level"]).to_csv(log_file, index=False)

# ---------------- Streamlit Sidebar ---------------- #

st.sidebar.title("🔐 IDS Navigation")
option = st.sidebar.radio("Select Feature", ("📊 Upload & Visualize", "📡 Real-Time Monitoring", "📝 Threat Report & Logging"))

# ---------------- Upload & Visualize ---------------- #

# ---------------- Upload & Visualize ---------------- #
if option == "📊 Upload & Visualize":
    st.title("🛡️ Intrusion Detection System - File Upload & Visualization")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Load CSV into DataFrame
            data = pd.read_csv(uploaded_file)
            st.success("✅ File successfully loaded!")
            
            # Preprocess the data for prediction
            X_input = preprocess_input(data)

            # Ensure X_input is a DataFrame
            if not isinstance(X_input, pd.DataFrame):
                X_input = pd.DataFrame(X_input)

            # Predict using the model
            predictions = model.predict(X_input)

            # Add predictions to the DataFrame
            X_input["Intrusion_Detected"] = ["Normal" if p == 0 else "Attack" for p in predictions.ravel()]

            # Display predictions
            st.write("### 🧾 Prediction Results")
            st.dataframe(X_input)
            
            # Visualize the traffic distribution
            st.write("### 📈 Traffic Distribution")
            fig1, ax1 = plt.subplots()
            ax1.pie([sum(predictions == 0), sum(predictions == 1)],
                    labels=["Normal", "Attack"],
                    colors=["green", "red"],
                    autopct="%1.1f%%")
            ax1.axis("equal")
            st.pyplot(fig1)

            # Attack Type Distribution if 'label' column exists
            if "label" in data.columns:
                attack_counts = data[X_input["Intrusion_Detected"] == "Attack"]["label"].value_counts()
                st.write("### 🧨 Attack Type Distribution")
                fig2, ax2 = plt.subplots()
                attack_counts.plot(kind="bar", color="crimson", ax=ax2)
                plt.xticks(rotation=45)
                plt.xlabel("Attack Type")
                plt.ylabel("Count")
                st.pyplot(fig2)

            # Feature Correlation Heatmap
            st.write("### 🔥 Feature Correlation Heatmap")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.heatmap(X_input.drop(columns=["Intrusion_Detected"]).corr(), cmap="coolwarm", ax=ax3)
            st.pyplot(fig3)

        except Exception as e:
            st.error(f"❌ Failed to load or process file: {e}")

# ---------------- Real-Time Monitoring ---------------- #
elif option == "📡 Real-Time Monitoring":
    st.title("📡 Real-Time Network Traffic Monitoring")
    placeholder = st.empty()

    for i in range(20):
        time.sleep(2)
        traffic_data = generate_fake_traffic()
        X_input = preprocess_input(traffic_data)
        prediction = model.predict(X_input)
        label = "Normal" if prediction[0] == 0 else "Attack"
        traffic_data["Intrusion_Detected"] = label

        with placeholder.container():
            st.write(f"📶 **Event {i+1}**")
            st.dataframe(traffic_data)

            if label == "Attack":
                st.error("⚠️ Intrusion Detected!")
                intrusion_details = f"""
                Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
                Source Bytes: {traffic_data.iloc[0]['src_bytes']}
                Destination Bytes: {traffic_data.iloc[0]['dst_bytes']}
                Duration: {traffic_data.iloc[0]['duration']}
                """
                try:
                    send_email_alert(intrusion_details)
                except Exception as e:
                    st.warning(f"⚠️ Email sending failed: {e}")

            fig, ax = plt.subplots()
            ax.pie([1 if label == "Normal" else 0, 1 if label == "Attack" else 0],
                   labels=["Normal", "Attack"],
                   colors=["green", "red"],
                   autopct="%1.1f%%")
            st.pyplot(fig)

    st.success("✅ Monitoring Completed.")

# ---------------- Threat Report & Logging ---------------- #
elif option == "📝 Threat Report & Logging":
    st.title("📑 Threat Logging & Report Generation")
    placeholder = st.empty()

    for i in range(20):
        time.sleep(2)
        traffic_data = generate_fake_traffic()
        X_input = preprocess_input(traffic_data)
        prediction = model.predict(X_input)
        label = "Normal" if prediction[0] == 0 else "Attack"

        new_entry = pd.DataFrame([{
            "Time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Duration": traffic_data.iloc[0]["duration"],
            "Protocol": traffic_data.iloc[0]["protocol_type"],
            "Service": traffic_data.iloc[0]["service"],
            "Flag": traffic_data.iloc[0]["flag"],
            "Src_Bytes": traffic_data.iloc[0]["src_bytes"],
            "Dst_Bytes": traffic_data.iloc[0]["dst_bytes"],
            "Threat_Level": label
        }])
        new_entry.to_csv(log_file, mode="a", header=False, index=False)

        with placeholder.container():
            st.write(f"📶 **Event {i+1}**")
            st.dataframe(new_entry)

            if label == "Attack":
                st.error("⚠️ Intrusion Detected and Logged!")

    st.success("✅ Threat Logging Completed!")

    st.write("### 📋 Complete Threat Log")
    threat_logs = pd.read_csv(log_file)
    st.dataframe(threat_logs)

    st.download_button("📥 Download Threat Report",
                       data=threat_logs.to_csv(index=False),
                       file_name="Threat_Report.csv",
                       mime="text/csv")

