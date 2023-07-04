import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import matplotlib.pyplot as plt

device ="cuda" if torch.cuda.is_available() else "cpu"

url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv"
data = pd.read_csv(url)


x = data["YearsExperience"]
y = data["Salary"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=87)
x_train = x_train.to_numpy().reshape(26,1)
x_test = x_test.to_numpy().reshape(-1,1)
y_train = y_train.to_numpy().reshape(-1,1)
y_test = y_test.to_numpy().reshape(-1,1)

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

class LinearRegressionModel2(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_layer = nn.Linear(in_features=1, out_features=1,dtype=torch.float64)

  def forward(self, x):
    return self.linear_layer(x)

torch.manual_seed(87)
model2 =LinearRegressionModel2()

cost_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params=model2.parameters(), lr=0.01)

# model.train() 告知模型我們在訓練階段
# model.eval() 告知模型我們在測試階段

epochs = 10000

train_cost_hist = []
test_cost_hist = []

for epoch in range(epochs):
    model2.train()
    y_pred = model2(x_train)
    train_cost = cost_fn(y_pred, y_train)
    
    train_cost_hist.append(train_cost.detach().numpy())
    
    optimizer.zero_grad() #optimizer梯度會疊加， 所以要歸零
    train_cost.backward() 
    #算出costfunction微分(requires_grade=True，會自動儲存到參數的梯度中(累加)
    optimizer.step() #開始優化(梯度下降，更新優化器內的參數，再傳入class參數中)
    
    model2.eval()
    with torch.inference_mode(): #在測試階段不需要追蹤梯度
        test_pred = model2(x_test)
        test_cost = cost_fn(test_pred, y_test)
        test_cost_hist.append(test_cost.detach().numpy())
    
    if epoch%1000 == 0:
       print(f"Epoch: {epoch:5}  train_cost: {train_cost: .4e}, test_cost: {test_cost: .4e}")      

plt.figure(figsize=(10,5),dpi=300)
plt.plot(range(0,10000), train_cost_hist, label="train_cost")
plt.plot(range(0,10000), test_cost_hist, label="test_cost")
plt.title("train & test cost")
plt.xlabel("epochs")
plt.ylabel("cost")
plt.legend()
plt.show()

print(model2.state_dict())

model2.eval()
with torch.inference_mode():
    y_pred = model2(x_test)

print(y_pred, y_test)

