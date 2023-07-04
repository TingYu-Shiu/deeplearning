import torch
import torch.nn as nn
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np

# 生成隨機數據集
X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=0)
X = StandardScaler().fit_transform(X)  # 標準化數據

# 轉換為Tensor
X_tensor = torch.Tensor(X)

# 定義自編碼器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(in_features=input_dim, out_features=50)
        self.decoder = nn.Linear(in_features=50, out_features=hidden_dim)

    def forward(self, x):
        encoded = self.encoder(x) #編碼器
        decoded = self.decoder(encoded) #解碼器
        return encoded, decoded

# 設置模型參數
input_dim = X_tensor.shape[1]
hidden_dim = 2  # 降維後的特徵維度

# 創建自編碼器模型
torch.manual_seed(87)
autoencoder = Autoencoder(input_dim, hidden_dim)

# 定義損失函數和優化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# 訓練自編碼器
num_epochs = 100
for epoch in range(num_epochs):
    encoded, decoded = autoencoder(X_tensor)
    loss = criterion(decoded, X_tensor) 
    #降維後，其loss很低，表示降維後的結果與原始資料所攜帶的特徵雷同

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# 獲得降維後的特徵
_,encoded_data = autoencoder(X_tensor)

# 使用聚類算法進行聚類（例如K-means）
from sklearn.cluster import KMeans

encoded_data = encoded_data.detach().numpy()

k = 3  # 聚類數量
kmeans = KMeans(n_clusters=k, random_state=0)
cluster_labels = kmeans.fit_predict(encoded_data)

# 輸出聚類結果
print(cluster_labels)

import matplotlib.pyplot as plt

# 繪製分群結果的散點圖
plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=cluster_labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustering Result')
plt.show()