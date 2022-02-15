import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

X_np, Y_np = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_np.astype(np.float32))
Y = torch.from_numpy(Y_np.astype(np.float32))

Y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape

input_size = n_features
output_size = 1

model = nn.Linear(input_size, 1)

learning_rate = 0.1
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_epochs = 100

for epoch in range(n_epochs):
    y_predicted = model.forward(X)

    loss = criterion(y_predicted, Y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch+1}, loss = {loss.item():.4f}')


predicted = model(X).detach().numpy()
plt.plot(X_np, Y_np, 'ro')
plt.plot(X_np, predicted, 'b')
plt.show()
