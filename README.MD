ICU Care Unit for Coma Patients 🚑
A real-time patient monitoring system that tracks heart rate, blood pressure, oxygen saturation, and respiratory rate using Flask, Machine Learning (RandomForest), and Matplotlib. It predicts critical conditions and sends email alerts to healthcare providers.

📌 Features
✅ Real-time vitals monitoring
✅ RandomForest-based alarm triggering
✅ ECG & vitals visualization (Matplotlib)
✅ Email alerts for critical conditions
✅ Flask-based Web UI for patient tracking

🛠️ Tech Stack
Backend: Flask, Pandas, NumPy

Machine Learning: RandomForest (sklearn)

Visualization: Matplotlib

Email Alerts: SMTP (smtplib)

Frontend: HTML, CSS (Flask templates)

📂 Project Structure
php
Copy
Edit
📁 care_unit_project  
│── app.py               # Main Flask app  
│── templates/  
│   │── index.html        # Dashboard UI  
│   │── plot.html         # Vitals visualization  
│   │── ecg_plot.html     # ECG visualization  
│── static/  
│   │── styles.css        # Styling files  
│   │── patient_vitals.png  # Vitals plots (generated dynamically)  
│── high_accuracy_refined_training_data.csv  # Training data for ML  
│── requirements.txt     # Python dependencies  
│── README.md            # Project documentation  
⚙️ Setup & Installation
1️⃣ Clone the Repository

git clone https://github.com/Harshithkumarkm2002/ICU-Care-Unit-for-Coma-Patients
cd icu-care-unit

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Flask Application
python app.py
The app will be available at http://127.0.0.1:5000/

🚀 Features Overview
🔹 Real-Time Monitoring
The system updates patient vitals periodically and detects unstable conditions.

🔹 Machine Learning for Alarm Triggering
Uses a RandomForest model trained on high_accuracy_refined_training_data.csv

Predicts if an alarm should be triggered based on patient vitals

🔹 Vitals & ECG Plots
Vitals Graph: /generate_plot/<patient_id>

ECG Simulation: /generate_ecg/<patient_id>

🔹 Email Alerts for Critical Patients
Automatically sends an alert if a patient's vitals go out of the normal range.

📝 Future Improvements
Integrate a real-time database (PostgreSQL, Firebase)

Use XGBoost or LightGBM for better ML performance

Add SMS alerts via Twilio
