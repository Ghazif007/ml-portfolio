import torch  
weight=torch.tensor(1.0,requires_grad=True)
X=torch.tensor(2.0)
target=torch.tensor(10.0)
learning_rate=0.25

for step in range (10):
    prediction=weight*X
    loss=(target - prediction)**2
    loss.backward()

    print("round",step,
          "weight:",round(weight.item(),2),
          "prediction:",round(prediction.item(),2),
          "loss:",round(loss.item(),2),
          "slope",round(weight.grad.item(),2))

    with torch.no_grad():
        weight-= learning_rate*weight.grad
       
    weight.grad.zero_()



