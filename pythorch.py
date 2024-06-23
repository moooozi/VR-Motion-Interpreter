import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import torch.onnx

# Load the data
data = pd.read_csv('2024011705-0110.csv')

# Preprocess the data
le = LabelEncoder()
data['step'] = le.fit_transform(data['step'])

# Split the data into features and target
features = data.drop('step', axis=1)
target = data['step']

# Split the data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Convert the training and testing sets into PyTorch tensors
featuresTrain = torch.Tensor(features_train.values)
targetTrain = torch.Tensor(target_train.values)
featuresTest = torch.Tensor(features_test.values)
targetTest = torch.Tensor(target_test.values)

# Create dataloaders for the training and testing sets
train = TensorDataset(featuresTrain, targetTrain)
test = TensorDataset(featuresTest, targetTest)

train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=64, shuffle=True)

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(37, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(50):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

# Switch to evaluation mode
model.eval()

correct = 0
total = 0

# No need to track gradients for validation, saves memory and computations
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)
        
        # Total number of labels
        total += labels.size(0)
        
        # Total correct predictions
        correct += (predicted == labels.long()).sum().item()

print('Accuracy of the network on the test data: %d %%' % (100 * correct / total))


# Save the trained model to ONNX
#dummy_input = Variable(torch.randn(1, 37))
#torch.onnx.export(model, dummy_input, "model.onnx")
