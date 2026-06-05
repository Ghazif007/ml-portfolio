import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
n = 1000

transaction_amount = np.random.uniform(100, 50000, n)
transaction_hour =np.random.randint(0,24,n)
distance_from_home = np.random.uniform(0,500,n)
is_foreign = np.random.randint(0,2,n)
daily_transactions=np.random.randint(1,20,n)

fraud = (
    (transaction_amount > 45000) |
    (distance_from_home > 450) |
    ((is_foreign == 1) & (transaction_amount > 40000)) |
    ((transaction_hour >= 1) & (transaction_hour <= 4) & 
     (transaction_amount > 30000))
).astype(int)

df = pd.DataFrame({
    "transaction_amount": transaction_amount,
    "transaction_hour": transaction_hour,
    "distance_from_home": distance_from_home,
    "is_foreign": is_foreign,
    "daily_transactions": daily_transactions,
    "fraud":fraud
})   


print(df.shape)
print(df.head())
print(df["fraud"].value_counts())
 
x = df.drop("fraud",axis=1)
y = df["fraud"]

x_train,x_test,y_train,y_test=train_test_split(
    x,y, test_size=0.2,random_state=42
)

print("train",len(x_train))
print("test",len(x_test))

model = GaussianNB()
model.fit(x_train,y_train)
y_pred =model.predict(x_test)

print("accuracy:",round(accuracy_score(y_test,y_pred),3))
print(classification_report(y_test, y_pred))

           