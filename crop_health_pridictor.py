import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

np.random.seed(42)
n = 1000

temperature = np.random.uniform(15, 45, n)
humidity = np.random.uniform(30, 90, n)
ph_level = np.random.uniform(4, 9, n)
sunlight_hours = np.random.uniform(2, 12, n)

healthy = (
    (temperature >= 18) & (temperature <= 38) &
    (humidity >= 45) &
    (ph_level >= 5.5) & (ph_level <= 8) &
    (sunlight_hours >= 4)
).astype(int)

df = pd.DataFrame({
    "temperature": temperature,
    "humidity": humidity,
    "ph_level": ph_level,
    "sunlight_hours": sunlight_hours,
    "healthy": healthy
})

X = df.drop("healthy", axis=1)
y = df["healthy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train:", len(X_train))
print("Test:", len(X_test))

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred))

param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 15]
}

grid_search=GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring="accuracy"


)  

grid_search.fit(X_train, y_train)
print("best k:",grid_search.best_params_)
print("best accuracy :",round(grid_search.best_score_,3))

best_knn= KNeighborsClassifier(n_neighbors=11)
best_knn.fit(X_train,y_train)
final_pres= best_knn.predict(X_test)

print("Final Accuracy:", round(accuracy_score(y_test, final_pres), 3))
print(classification_report(y_test,final_pres))


