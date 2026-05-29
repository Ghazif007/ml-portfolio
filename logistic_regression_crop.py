import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV

np.random.seed(42)
n=1000

temperature= np.random.uniform(15,45,n)
humidity=np.random.uniform(30,90,n)
ph_level =np.random.uniform(4,9,n)
sunlight_hours=np.random.uniform(2,12,n)

healthy =(
    (temperature >= 18) & (temperature <= 38)&
    (humidity >=45)&
    (ph_level >=5.5) & (ph_level <=8)&
    (sunlight_hours >= 4)
    
).astype(int)

df= pd.DataFrame({
    "temperature":temperature,
    "humidity":humidity,
    "ph_level":ph_level,
    "sunlight_hours":sunlight_hours,
    "healthy":healthy

})

x=df.drop("healthy",axis=1)
y=df["healthy"]

x_train,x_test,y_train,y_test=train_test_split(
    x,y, test_size=0.2 ,random_state=42
)


print("Train:", len(x_train))
print("Test:", len(x_test))

model=LogisticRegression(random_state=42,class_weight="balanced")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("accuracy",round(accuracy_score(y_test,y_pred),3))
print(classification_report(y_test,y_pred))

param_grid= {
    "C": [0.01,0.1,1,10,100],
    "class_weight": ["balanced",None]
}

grid_search =GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=5,
    scoring="f1"
)

grid_search.fit(x_train, y_train)

print("best parameters",grid_search.best_params_)
print("best f1 score:",round(grid_search.best_score_,3))

best_model = LogisticRegression(
    C=0.1,
    class_weight= "balanced",
    random_state=42
)

best_model.fit(x_train,y_train)
final_pred=best_model.predict(x_test)


print("Final Accuracy:", round(accuracy_score(y_test, final_pred), 3))
print(classification_report(y_test, final_pred))
