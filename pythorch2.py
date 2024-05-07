from collections import Counter
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import glob

print("PyTorch version:", torch.__version__)
# Get a list of all CSV files in the "MLTrainingData/" directory
csv_files = glob.glob(os.path.join("Assets/MLTrainingData/", "*.csv"))

dfs = []

# Loop through the list of CSV files
for csv_file in csv_files:
    # Read each CSV file into a DataFrame and add it to the list
    df = pd.read_csv(csv_file)
    dfs.append(df)

# Concatenate all the DataFrames in the list into one DataFrame
data = pd.concat(dfs, ignore_index=True)

# Preprocess data
data['step'].fillna('none', inplace=True)
data['step'] = data['step'].map({'none': 0, 'left': 1, 'right': 2})
data = data.dropna()

# Calculate relative positions
data['RelPosLHandHeadX'] = data['cPosLHX'] - data['cPosHeadX']
data['RelPosLHandHeadY'] = data['cPosLHY'] - data['cPosHeadY']
data['RelPosLHandHeadZ'] = data['cPosLHZ'] - data['cPosHeadZ']

data['RelPosRHandHeadX'] = data['cPosRHX'] - data['cPosHeadX']
data['RelPosRHandHeadY'] = data['cPosRHY'] - data['cPosHeadY']
data['RelPosRHandHeadZ'] = data['cPosRHZ'] - data['cPosHeadZ']


column_names = ['dPosLHX', 'dPosLHY', 'dPosLHZ', 
                'dRotLHX', 'dRotLHY', 'dRotLHZ', 
                'dPosRHX', 'dPosRHY', 'dPosRHZ', 
                'dRotRHX', 'dRotRHY', 'dRotRHZ',
                'dPosHeadX', 'dPosHeadY', 'dPosHeadZ', 
                'dRotHeadX', 'dRotHeadY', 'dRotHeadZ', 
                'RelPosLHandHeadX', 'RelPosLHandHeadY', 'RelPosLHandHeadZ',
                'RelPosRHandHeadX', 'RelPosRHandHeadY', 'RelPosRHandHeadZ']

'''
# Normalize data
scaler = MinMaxScaler()
data[column_names] = scaler.fit_transform(data[column_names])
'''

# Create a new training set with only the relevant attributes
training_data = data[column_names].copy()

# Create sequences and labels
sequences = []
labels = []
for i in range(0, len(training_data) - 6):
    sequences.append(training_data.iloc[i:i+7].values) 
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
print(f"Total number of entries: {len(training_data)}")
print(f"Number of 'none' sequences: {label_counts[0]}")
print(f"Number of 'left' sequences: {label_counts[1]}")
print(f"Number of 'right' sequences: {label_counts[2]}")

# Split data into training set and temporary set using 80-20 split
X_temp, X_test, y_temp, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Split the temporary set into validation set and final training set using 80-20 split
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.long)

'''
# Split data
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.long)
'''
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

model = LSTM(input_size=24, hidden_size=50, num_layers=2, num_classes=3)

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float("Inf") 
epochs_no_improve = 0
n_epochs_stop = 5

for epoch in range(1000):  # Increase to 1000 epochs
    # Training
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluation on validation set
    model.eval()
    with torch.no_grad():
        outputs_val = model(X_val)
        val_loss = criterion(outputs_val, y_val)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
    else:
        epochs_no_improve += 1
        if epochs_no_improve == n_epochs_stop:
            print('Optimal fitting reached. Stopping!')
            break




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