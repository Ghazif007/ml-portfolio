import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
np.random.seed(42)
n = 1000

quantity = np.random.randint(1, 50, n)
is_premium = np.random.randint(0, 2, n)
is_discount = np.random.randint(0, 2, n)
category = np.random.choice(["A", "B", "C"], n)

price = (
    3000
    + 40 * quantity
    + 800 * is_premium
    - 500 * is_discount
    + np.random.normal(0, 600, n)
)

df = pd.DataFrame({
    "quantity": quantity,
    "is_premium": is_premium,
    "is_discount": is_discount,
    "category": category,
    "price": price
})

df=pd.get_dummies(df,columns=["category"],drop_first=True)
df = df.astype({col: int for col in ["category_B", "category_C"]})

X = df.drop("price", axis=1)  # ✅
y = df["price"]

from sklearn.model_selection import train_test_split

# First split - separate train from the rest
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Second split - separate val and test from the rest
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

val_preds = model.predict(X_val)
print("Val MAE:", round(mean_absolute_error(y_val, val_preds), 2))
print("Val R2:", round(r2_score(y_val, val_preds), 3))

param_grid = {
    "n_estimators":[50,100,200],
    "max_depth":[3,5,8,None]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="r2"

)
grid_search.fit(X_train, y_train) 
print("Best Parameters:", grid_search.best_params_)
print("Best R2 Score:", round(grid_search.best_score_, 3))

best_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

best_model.fit(X_train, y_train)
test_preds = best_model.predict(X_test)

print("Test MAE:", round(mean_absolute_error(y_test, test_preds), 2))
print("Test R2:", round(r2_score(y_test, test_preds), 3))  