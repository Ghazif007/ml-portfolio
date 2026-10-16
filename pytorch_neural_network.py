import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.optim.adam

np.random.seed(42)
torch.manual_seed(42)
n=500

study_hours = np.random.uniform(0,10,n)
sleep_hours = np.random.uniform(0,10,n)
attendance = np.random.uniform(40,100,n)

passed = ((study_hours > 4) & (attendance > 60) & (sleep_hours > 5)).astype(int)

df=pd.DataFrame({
    "study_hours":study_hours,
    "sleep_hours":sleep_hours,
    "attendence":attendance,
    "passed":passed
})

X=df.drop("passed",axis=1)
Y=df["passed"]

X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,test_size=0.2,random_state=42
)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

X_train=torch.tensor(X_train,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)
Y_train=torch.tensor(Y_train.values, dtype=torch.float32).view(-1,1)
Y_test = torch.tensor(Y_test.values, dtype=torch.float32).view(-1, 1)

model =nn.Sequential(
    nn.Linear(3,8),
    nn.ReLU(),
    nn.Linear(8,4),
    nn.ReLU(),
    nn.Linear(4,1),
    nn.Sigmoid()

)

loss_fn = nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)


for epoch in range(300):
    prediction=model(X_train)
    loss=loss_fn(prediction,Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50==0:
        print("epoch",epoch,"|loss:",round(loss.item(),4))


with torch.no_grad():
        test_probs= model(X_test)
        test_answer=(test_probs > 0.5).float()
        accuracy= (test_answer==Y_test).float().mean()

print("pytorch accuracy",round(accuracy.item(),3))

        
    

