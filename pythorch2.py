from collections import Counter
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('2024011705-0110.csv')

# Preprocess data
data['step'].fillna('none', inplace=True)
data['step'] = data['step'].map({'none': 0, 'left': 1, 'right': 2})
scaler = MinMaxScaler()
data.iloc[:, 2:20] = scaler.fit_transform(data.iloc[:, 2:20])  # Only consider dPos and dRot values

# Create sequences and labels
sequences = []
labels = []
for i in range(0, len(data) - 6):
    sequences.append(data.iloc[i:i+7, 2:20].values)  # Only consider dPos and dRot values
    steps = data.iloc[i+2:i+5]['step']
    if 1 in steps.values:
        labels.append(1)
    elif 2 in steps.values:
        labels.append(2)
    else:
        labels.append(0)

# Count the occurrences of each label
label_counts = Counter(labels)

# Print the total number of sequences and entries
print(f"Total number of sequences: {len(sequences)}")
print(f"Total number of entries: {len(data)}")
print(f"Number of 'none' sequences: {label_counts[0]}")
print(f"Number of 'left' sequences: {label_counts[1]}")
print(f"Number of 'right' sequences: {label_counts[2]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(input_size=18, hidden_size=50, num_layers=2, num_classes=3)

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Evaluate model
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    total = y_test.size(0)
    correct = (predicted == y_test).sum().item()

    # Calculate overall accuracy
    overall_accuracy = correct / total

    # Calculate accuracy for "left", "right" and "none" separately
    left_indices = (y_test == 1)
    right_indices = (y_test == 2)
    none_indices = (y_test == 0)

    left_correct = (predicted[left_indices] == y_test[left_indices]).sum().item()
    right_correct = (predicted[right_indices] == y_test[right_indices]).sum().item()
    none_correct = (predicted[none_indices] == y_test[none_indices]).sum().item()

    left_accuracy = left_correct / left_indices.sum().item()
    right_accuracy = right_correct / right_indices.sum().item()
    none_accuracy = none_correct / none_indices.sum().item()

print(f'Overall Accuracy: {overall_accuracy*100}%')
print(f'Left Accuracy: {left_accuracy*100}%')
print(f'Right Accuracy: {right_accuracy*100}%')
print(f'None Accuracy: {none_accuracy*100}%')