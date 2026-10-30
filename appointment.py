import pandas as pd
import os

DB_PATH = "database/appointments.csv"

def save_appointment(patient, age, gender, doctor, date, time):
    os.makedirs("database", exist_ok=True)

    new_data = pd.DataFrame([{
        "Patient": patient,
        "Age": age,
        "Gender": gender,
        "Doctor": doctor,
        "Date": str(date),
        "Time": time,
        "Status": "Booked"
    }])

    if os.path.exists(DB_PATH):
        old_data = pd.read_csv(DB_PATH)
        data = pd.concat([old_data, new_data], ignore_index=True)
    else:
        data = new_data

    data.to_csv(DB_PATH, index=False)

def load_appointments():
    if os.path.exists(DB_PATH):
        return pd.read_csv(DB_PATH)
    else:
        return pd.DataFrame(columns=[
            "Patient",
            "Age",
            "Gender",
            "Doctor",
            "Date",
            "Time",
            "Status"
        ])