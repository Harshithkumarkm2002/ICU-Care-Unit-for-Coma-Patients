from flask import Flask, render_template, jsonify, request, redirect
import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import smtplib
from email.mime.text import MIMEText
import random
from threading import Thread


app = Flask(__name__, 
            template_folder='C:/Users/harshith/OneDrive/Desktop/projects/careee/templates',  # Custom template folder path
            static_folder='C:/Users/harshith/OneDrive/Desktop/projects/careee/static')  # Static files folder path

# Create a synthetic initial dataset
initial_data = {
    'patient_id': [f'P{i+1}' for i in range(15)],  # 15 patients
    'heart_rate': np.random.randint(60, 120, 15),
    'systolic_bp': np.random.randint(90, 140, 15),
    'diastolic_bp': np.random.randint(60, 90, 15),
    'oxygen_saturation': np.random.uniform(85, 100, 15),
    'respiratory_rate': np.random.randint(12, 25, 15),
    'alarm_triggered': np.random.choice([0, 1], 15, p=[0.8, 0.2])
}


df = pd.DataFrame(initial_data)

train_data= pd.read_csv("high_accuracy_refined_training_data.csv")
dff=pd.DataFrame(train_data)

# Update feature set for machine learning
X = dff.drop(['patient_id', 'alarm_triggered'], axis=1)
y = dff['alarm_triggered']

# Machine Learning Model for Alarm Prediction
X = dff.drop(['patient_id', 'alarm_triggered'], axis=1)
y = dff['alarm_triggered']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# Email Alerts (SMTP)
smtp_server = 'smtp.gmail.com'
smtp_port = 587
sender_email = 'harshithk288@gmail.com'
sender_password = 'srcb ethb whmq udqn'
recipient_email = 'harshithk288@gmail.com'
def send_email_alert(patient_id, details):
    subject = f"Critical Alert for Patient {patient_id}"
    body = f"Critical condition detected for Patient {patient_id}: {details}. Immediate attention required!"
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print(f"Email Alert Sent for Patient {patient_id}")
    except Exception as e:
        print(f"Error sending email: {e}")

        Thread(target=send_email_alert).start()



# Update patient data to simulate real-time changes
def update_patient_data(df):
    for col in ['heart_rate', 'systolic_bp', 'diastolic_bp', 'oxygen_saturation', 'respiratory_rate']:
        df[col] += np.random.randint(-5, 6, size=len(df))

    num_stable = int(len(df) * 0.8)
    stable_patient_indices = random.sample(range(len(df)), k=num_stable)

    for i in stable_patient_indices:
        df.loc[i, 'heart_rate'] = random.randint(60, 100)
        df.loc[i, 'systolic_bp'] = random.randint(90, 120)
        df.loc[i, 'diastolic_bp'] = random.randint(60, 80)
        df.loc[i, 'oxygen_saturation'] = random.uniform(95, 100)
        df.loc[i, 'respiratory_rate'] = random.randint(12, 20)

    unstable_patient_indices = [i for i in range(len(df)) if i not in stable_patient_indices]
    for i in unstable_patient_indices:
        df.loc[i, 'heart_rate'] = random.randint(120, 180)
        df.loc[i, 'systolic_bp'] = random.randint(140, 200)
        df.loc[i, 'diastolic_bp'] = random.randint(90, 120)
        df.loc[i, 'oxygen_saturation'] = random.uniform(80, 94)
        df.loc[i, 'respiratory_rate'] = random.randint(25, 40)

    return df

# Update stability check to include systolic and diastolic BP thresholds
def check_stability(row):
    stable_heart_rate = 60 <= row['heart_rate'] <= 100
    stable_systolic_bp = 90 <= row['systolic_bp'] <= 120
    stable_diastolic_bp = 60 <= row['diastolic_bp'] <= 80
    stable_o2 = 95 <= row['oxygen_saturation'] <= 100
    stable_rr = 12 <= row['respiratory_rate'] <= 20

    return stable_heart_rate and stable_systolic_bp and stable_diastolic_bp and stable_o2 and stable_rr

# Update routes to display and plot systolic and diastolic blood pressure
@app.route('/')
def index():
    global df
    df = update_patient_data(df)

    patient_data = []
    for _, row in df.iterrows():
        patient_status = "✅ Patient is stable." if check_stability(row) else "⚠️ CRITICAL CONDITION DETECTED! ALARM TRIGGERED! ⚠️"
        if not check_stability(row):
            send_email_alert(row['patient_id'], row)
        patient_data.append({
            'patient_id': row['patient_id'],
            'heart_rate': row['heart_rate'],
            'systolic_bp': row['systolic_bp'],
            'diastolic_bp': row['diastolic_bp'],
            'oxygen_saturation': row['oxygen_saturation'],
            'respiratory_rate': row['respiratory_rate'],
            'status': patient_status,
            'plot_link': f'/generate_plot/{row["patient_id"]}',
            'ecg_link': f'/generate_ecg/{row["patient_id"]}'  # Link to ECG plot
        })
    
    return render_template('index.html', patients=patient_data)

@app.route('/generate_plot/<patient_id>')
def generate_plot(patient_id):
    global df
    patient_data = df[df['patient_id'] == patient_id].iloc[0]

    plt.figure(figsize=(10, 6))
    categories = ['Heart Rate', 'Systolic BP', 'Diastolic BP', 'Oxygen Saturation', 'Respiratory Rate']
    values = [patient_data['heart_rate'], patient_data['systolic_bp'], patient_data['diastolic_bp'],
              patient_data['oxygen_saturation'], patient_data['respiratory_rate']]
    plt.bar(categories, values, color='skyblue', edgecolor='black')
    plt.title(f"Patient {patient_id} Vitals", fontsize=18)
    plt.ylabel('Values', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    img_path = os.path.join(app.static_folder, f'{patient_id}_vitals.png')
    plt.savefig(img_path)
    plt.close()

    return redirect(f'/view_plot/{patient_id}')

# View plot route
@app.route('/view_plot/<patient_id>')
def view_plot(patient_id):
    img_path = f'/static/{patient_id}_vitals.png'
    return render_template('plot.html', patient_id=patient_id, img_path=img_path)

# Generate ECG plot (simulated)
@app.route('/generate_ecg/<patient_id>')
def generate_ecg(patient_id):
    # Generate a simulated ECG-like plot
    t = np.linspace(0, 10, 1000)
    ecg_signal = np.sin(2 * np.pi * 1 * t) + 0.2 * np.random.randn(t.shape[0])  # Simulated ECG signal

    plt.figure(figsize=(10, 6))
    plt.plot(t, ecg_signal, label="ECG Signal", color='blue')
    plt.title(f"ECG for Patient {patient_id}", fontsize=18)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()  


    img_path = os.path.join(app.static_folder, f'{patient_id}_ecg.png')
    plt.savefig(img_path)
    plt.close()

    return redirect(f'/view_ecg/{patient_id}')

# View ECG plot
@app.route('/view_ecg/<patient_id>')
def view_ecg(patient_id):
    img_path = f'/static/{patient_id}_ecg.png'
    return render_template('ecg_plot.html', patient_id=patient_id, img_path=img_path)

# Update patient data (simulate real-time changes)
@app.route('/update', methods=['POST'])
def update_patient():
    global df
    df = update_patient_data(df)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
 