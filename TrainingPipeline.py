import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

n_samples, n_features = X.shape
print(X.shape)

X_test = torch.tensor([5], dtype=torch.float32)
input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

lr = 0.01
epochs = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(epochs):
    y_pred = model(X)

    l = loss(y_pred, Y)

    l.backward() # dLoss/dw

    # Perform a weight optimization/update step.
    optimizer.step()
    # We need to reset grads
    optimizer.zero_grad()

    if epoch % 2 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.3f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')