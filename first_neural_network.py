import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

n= 500

study_hours = np.random.uniform(0, 10, n)
sleep_hours = np.random.uniform(3, 10, n)
attendance = np.random.uniform(40, 100, n)

passed = (
    (study_hours > 4) & (attendance > 60) & (sleep_hours > 5)
).astype(int)

df = pd.DataFrame({
    "study_hours": study_hours,
    "sleep_hours": sleep_hours,
    "attendance": attendance,
    "passed": passed
})

X=df.drop("passed",axis=1)
Y=df["passed"]

X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y, test_size=0.2 , random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)    # blank 1: learn + convert
X_test = scaler.transform(X_test)      # blank 2: only convert

# Step 2: Build the Neural Network
model = MLPClassifier(
    hidden_layer_sizes=(8, 4),   # 2 hidden layers: 8 neurons, then 4
    learning_rate_init=0.01,     # spoon size 🥄
    max_iter=1000,               # max chai attempts ☕
    random_state=42
)

# Step 3: Train, predict, score
model.fit(X_train, Y_train)              # blanks 3 & 4
y_pred = model.predict(X_test)      # blank 5

print("Neural Network Accuracy:", round(accuracy_score(Y_test, y_pred), 3))