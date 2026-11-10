import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(7)
n=800

price = np.random.uniform(1500, 15000, n)
delivery_days = np.random.randint(1, 11, n)
rating = np.random.uniform(2.5, 5.0, n)
first_time_buyer = np.random.randint(0, 2, n)

returned = (((delivery_days > 6) & (rating < 4)) |
            ((first_time_buyer == 1) & (price > 10000))).astype(int)
flip = np.random.rand(n) < 0.10
returned[flip] = 1 - returned[flip]

X = np.column_stack([price, delivery_days, rating, first_time_buyer])
y = returned.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mean=X_train.mean(axis=0)
std=X_train.std(axis=0)

X_train= (X_train-mean)/std
X_test=(X_test-mean)/std

def sigmoid(z):
    return 1/(1+np.exp(-z))

def relu(z):
    return np.maximum(0,z)

np.random.seed(0)
W1 = np.random.randn(4, 8) * 0.5
b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 1) * 0.5
b2 = 0.0

lr = 0.5
m = len(X_train)

for epoch in range(2000):
    # ---------- FORWARD ➡️ ----------
    z1 = X_train @ W1 + b1
    h = relu(z1)
    z2 = h @ W2 + b2
    p = sigmoid(z2)

    loss = -np.mean(y_train * np.log(p + 1e-9) + (1 - y_train) * np.log(1 - p + 1e-9))

    # ---------- BACKWARD ⬅️ ----------
    error = p - y_train                         # manager's mistake
    dW2 = h.T @ error / m                     # blank 1: who does the manager listen to?
    db2 = error.mean()

    dh = error @ W2.T                          # blank 2: pass blame back through the manager's weights
    dz1 = dh * (z1 > 0)                       # blank 3: only staff who SPOKE UP get blamed
    dW1 = X_train.T @ dz1 / m
    db1 = dz1.mean(axis=0, keepdims=True)

    # ---------- FIX ⛰️ ----------
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    W1 = W1 - lr * dW1                          # blank 4
    b1 = b1 - lr * db1

    if epoch % 400 == 0:
        print("Epoch", epoch, "| loss:", round(loss, 4))

# ---------- TEST 📝 ----------
h_test = relu(X_test @ W1 + b1)
p_test = sigmoid(h_test @ W2 + b2)
accuracy = ((p_test > 0.5) == y_test).mean()
print("Hidden-layer accuracy:", round(accuracy, 3))
print("(single neuron got 0.762, Random Forest got 0.887)")
