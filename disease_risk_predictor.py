import numpy as np
import pandas as pd 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(42)
n=1000

age=np.random.randint(25,80,n)
systolic_bp = np.random.uniform(90,200,n)
diastolic_bp=np.random.uniform(60,130,n)
sugar_level=np.random.uniform(70,300,n)
bmi=np.random.uniform(15,45,n)
cholesterol=np.random.uniform(100,300,n)
smoking=np.random.randint(0,2,n)
sgpt=np.random.uniform(7,150,n)
sgot=np.random.uniform(7,150,n)

high_risk = (
    ((systolic_bp > 180) & (sugar_level > 250)) |
    ((diastolic_bp > 110) & (cholesterol > 270)) |
    ((bmi > 38) & (age > 55)) |
    ((smoking == 1) & (age > 60) & (cholesterol > 260)) |
    ((sgpt > 130) & (sgot > 120))
).astype(int)

medium_risk = (
    ((systolic_bp > 160) & (bmi > 28)) |
    ((diastolic_bp > 100) & (age > 40)) |
    ((sugar_level > 220) & (cholesterol > 220)) |
    ((smoking == 1) & (bmi > 28)) |
    ((sgpt > 110) & (sgot > 80))
).astype(int)

risk = np.where(high_risk == 1, 2,
       np.where(medium_risk == 1, 1, 0))

df=pd.DataFrame({
    "age":age,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "sugar_level": sugar_level,
    "bmi": bmi,
    "cholesterol": cholesterol,
    "smoking": smoking,
    "sgpt": sgpt,
    "sgot": sgot,
    "risk": risk
})
print(df.shape)
print(df.head())
print(df["risk"].value_counts())

x=df.drop("risk",axis=1)
y=df["risk"]

x_train,x_test,y_train,y_test=train_test_split(
    x,y, test_size=0.2 ,random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

model = SVC(kernel="rbf",random_state=42)
model.fit(X_train,y_train)

model_pred=model.predict(X_test)

print("accuracy",round(accuracy_score(y_test,model_pred),3))
print("classification",classification_report(y_test,model_pred))

param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["rbf", "linear"],
    "class_weight": ["balanced", None]
}


grid_search= GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=3,
    scoring="f1_macro"
)
grid_search.fit(X_train,y_train)

print("best parameter",grid_search.best_params_)
print("best score",round(grid_search.best_score_,3))

best_model = SVC(
    C=10,
    kernel="rbf",
    class_weight=None,
    random_state=42
)

best_model.fit(X_train, y_train)
final_pred = best_model.predict(X_test)

print("Final Accuracy:", round(accuracy_score(y_test, final_pred), 3))
print(classification_report(y_test, final_pred))