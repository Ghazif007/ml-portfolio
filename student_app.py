import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def train_model():
    np.random.seed(42)
    torch.manual_seed(42)
    n = 500

    study_hours = np.random.uniform(0, 10, n)
    sleep_hours = np.random.uniform(3, 10, n)
    attendance = np.random.uniform(40, 100, n)
    passed = ((study_hours > 4) & (attendance > 60) & (sleep_hours > 5)).astype(int)

    df = pd.DataFrame({"study_hours": study_hours, "sleep_hours": sleep_hours,
                       "attendance": attendance, "passed": passed})

    X = df.drop("passed", axis=1)
    y = df["passed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    model = nn.Sequential(
        nn.Linear(3, 8), nn.ReLU(),
        nn.Linear(8, 4), nn.ReLU(),
        nn.Linear(4, 1), nn.Sigmoid()
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(300):
        prediction = model(X_train)
        loss = loss_fn(prediction, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, scaler


model, scaler = train_model()

st.title("student pass pridector")
st.write("move the slidersand my nural network will pridict if the student passes")

study = st.slider("study hours perday",0.0,10.0,0.5)
sleep =st.slider("sleep hours per night",3.0,10.0,7.0)
attend = st.slider("Attendence (%)",40.0,100.0,75.0)

if st.button("predict"):
    student = pd.DataFrame({"study_hours": [study],
                        "sleep_hours": [sleep],
                        "attendance": [attend]})


    student_scaled=scaler.transform(student)
    with torch.no_grad():
        prob=model(torch.tensor(student_scaled,dtype=torch.float32)).item()
        if prob > 0.5:
            st.success(f"PASS ✅ — {prob * 100:.1f}% sure")
        else:
            st.error(f"FAIL ❌ — {(1 - prob) * 100:.1f}% sure")

            




