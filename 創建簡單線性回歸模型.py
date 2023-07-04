import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv"
data = pd.read_csv(url)


x = data["YearsExperience"]
y = data["Salary"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=87)
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)


class LinearRegressionModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.w = nn.Parameter(torch.rand(1, requires_grad=True))
    self.b = nn.Parameter(torch.rand(1, requires_grad=True))

  def forward(self, x):
    return self.w*x + self.b

torch.manual_seed(87)
model = LinearRegressionModel()
# list(model.parameters())
model.state_dict()

cost_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

# model.train() 告知模型我們在訓練階段
# model.eval() 告知模型我們在測試階段

epochs = 10000

train_cost_hist = []
test_cost_hist = []

for epoch in range(epochs):
    model.train()
    y_pred = model(x_train)
    train_cost = cost_fn(y_pred, y_train)
    
    train_cost_hist.append(train_cost.detach().numpy())
    
    optimizer.zero_grad() #optimizer梯度會疊加， 所以要歸零
    train_cost.backward() 
    #算出costfunction微分(requires_grade=True，會自動儲存到參數的梯度中(累加)
    optimizer.step() #開始優化(梯度下降，更新優化器內的參數，再傳入class參數中)
    
    model.eval()
    with torch.inference_mode(): #在測試階段不需要追蹤梯度
        test_pred = model(x_test)
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

print(model.state_dict())

model.eval()
with torch.inference_mode():
    y_pred = model(x_test)

print(y_pred, y_test)

torch.save(obj=model.state_dict(),f='model/pytorch_linear_regression.pt')

#讀取之前訓練好的模型參數
model_1 = LinearRegressionModel()
model_1.load_state_dict(torch.load(f='model/pytorch_linear_regression.pt'))

#----------------LinearRegressionModle2------------------

class LinearRegressionModel2(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_layer = nn.Linear(in_features=1, out_features=1)
    
    #self.w = nn.Parameter(torch.rand(1, requires_grad=True))
    #self.b = nn.Parameter(torch.rand(1, requires_grad=True))

  def forward(self, x):
    return self.linear_layer(x)

torch.manual_seed(87)
model2 =LinearRegressionModel2()

#https://colab.research.google.com/drive/1bK0k_0eQy3bpTGDErAFIgvwulUN8e1Im?hl=zh-tw#scrollTo=rmtcaIb-h6qw

