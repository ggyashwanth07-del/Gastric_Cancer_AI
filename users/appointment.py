import pandas as pd
import os

DB = "database/appointments.csv"

def save_appointment(patient, age, gender, doctor, date, time):

    new = pd.DataFrame([{
        "Patient": patient,
        "Age": age,
        "Gender": gender,
        "Doctor": doctor,
        "Date": str(date),
        "Time": time,
        "Status": "Booked"
    }])

    if os.path.exists(DB):
        old = pd.read_csv(DB)
        new = pd.concat([old, new], ignore_index=True)

    new.to_csv(DB, index=False)

def load_appointments():
    if os.path.exists(DB):
        return pd.read_csv(DB)
    return pd.DataFrame()