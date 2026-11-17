# ml-portfolio

My machine learning learning journey: projects and practice code, from classic ML models to neural networks in PyTorch.

## 🌐 Live Demo

**Student Pass Predictor**, a PyTorch neural network you can try in your browser:
👉 https://ghazif-pass-predictor.streamlit.app/

## Projects

Listed in the order I built them.

| # | Project | What it predicts | Model | Result | File |
|---|---|---|---|---|---|
| 1 | Sales Price Prediction | Price of a sale from quantity, premium and discount | Random Forest + Grid Search | R² 0.51 on test data | `sales_price_prediction.py` |
| 2 | Crop Health (KNN) | Whether a crop is healthy from temperature, humidity, soil pH and sunlight | K-Nearest Neighbours, best k found with Grid Search | 81.5% accuracy | `crop_health_pridictor.py` |
| 3 | Crop Health (Logistic Regression) | Same problem as #2, to compare models | Logistic Regression + Grid Search | 67.5% accuracy | `logistic_regression_crop.py` |
| 4 | Fraud Detection | Whether a card transaction is fraud | Gaussian Naive Bayes | 87.5% accuracy | `fraud_detection.py` |
| 5 | Disease Risk | Low, medium or high health risk from age, blood pressure, sugar, BMI and liver values | SVM + Grid Search | 77% accuracy | `disease_risk_predictor.py` |
| 6 | First Neural Network | Whether a student passes from study, sleep and attendance | scikit-learn MLP (2 hidden layers) | 98% accuracy | `first_neural_network.py` |
| 7 | Gradient Descent in PyTorch | Learns one weight step by step, showing how training works | PyTorch autograd | Learning exercise | `first_pytorch.py` |
| 8 | Neural Network in PyTorch | Student pass/fail, rebuilt in PyTorch | 3-layer network, Adam optimizer | See live demo | `pytorch_neural_network.py` |
| 9 | Streamlit App | Web app around project #8 | PyTorch + Streamlit | Live demo above | `student_app.py` |
| 10 | Neural Network from Scratch | Whether an online order gets returned, from price, delivery days, rating and first-time buyer | NumPy only, with hand-written backpropagation | 86.9% accuracy | `numpy_hidden_layer.py` |

## What I learned along the way

- Splitting data into train, validation and test sets
- Tuning models with Grid Search and cross-validation
- Scaling features, and why distance-based models like KNN and SVM need it
- Comparing models on the same problem (projects 2 and 3)
- How a neural network learns: forward pass, loss, backpropagation and weight updates, first by hand in NumPy, then with PyTorch
- Deploying a model as a web app with Streamlit

## About the data

All datasets are generated in code with NumPy using simple rules, so each project can focus on the modelling. Results show how well each model learned those rules, not performance on real-world data.

## Tools

Python · NumPy · Pandas · scikit-learn · PyTorch · Streamlit

## How to run

```bash
pip install -r requirements.txt
python fraud_detection.py              # any project file
streamlit run student_app.py           # the web app
```
