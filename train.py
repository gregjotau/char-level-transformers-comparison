import torch
import torch.nn as nn
import torch.optim as optim
import time


# Simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


# Generate dummy data
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))

# Initialize model
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
print(f"Starting training on device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
start_time = time.time()

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}')

print(f"Training completed in {time.time() - start_time:.2f} seconds")