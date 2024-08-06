from collections import Counter
from datetime import datetime
import random
from sklearn.utils import compute_class_weight
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import glob

seed = 42  # Choose a seed

# Set the random seed for numpy, torch and python's built-in random module
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

print("PyTorch version:", torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get a list of all CSV files in the "MLTrainingData/" directory
csv_files = glob.glob(os.path.join("Assets/MLTrainingData/", "*.csv"))

# Initialize lists to store sequences and labels
sequences = []
labels = []
sequences_eval = []
labels_eval = []

# Loop through the list of CSV files
for csv_file in csv_files:
    # Read each CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Preprocess data
    df = df.dropna()

    # Calculate relative positions
    df['RelPosLHandHeadX'] = df['cPosLHX'] - df['cPosHeadX']
    df['RelPosLHandHeadY'] = df['cPosLHY'] - df['cPosHeadY']
    df['RelPosLHandHeadZ'] = df['cPosLHZ'] - df['cPosHeadZ']

    df['RelPosRHandHeadX'] = df['cPosRHX'] - df['cPosHeadX']
    df['RelPosRHandHeadY'] = df['cPosRHY'] - df['cPosHeadY']
    df['RelPosRHandHeadZ'] = df['cPosRHZ'] - df['cPosHeadZ']

    sequenceFeatures = ['dPosLHX', 'dPosLHY', 'dPosLHZ', 
                    'dRotLHX', 'dRotLHY', 'dRotLHZ', 
                    'dPosRHX', 'dPosRHY', 'dPosRHZ', 
                    'dRotRHX', 'dRotRHY', 'dRotRHZ',
                    'dPosHeadX', 'dPosHeadY', 'dPosHeadZ', 
                    'dRotHeadX', 'dRotHeadY', 'dRotHeadZ', 
                    'RelPosLHandHeadX', 'RelPosLHandHeadY', 'RelPosLHandHeadZ',
                    'RelPosRHandHeadX', 'RelPosRHandHeadY', 'RelPosRHandHeadZ']

    # Create a new training set with only the relevant attributes
    training_data = df[sequenceFeatures].copy()
    filename = os.path.basename(csv_file)
    # Create sequences and labels for each file
    for i in range(0, len(training_data) - 6):
        sequence = training_data.iloc[i:i+7].values
        steps = df.iloc[i+2:i+5]['step']
        if 1 in steps.values:
            label = 1
        elif 2 in steps.values:
            label = 2
        else:
            label = 0

        # If the CSV file name starts with 'eval_', append to evaluation sets
        if filename.startswith('eval_'):
            sequences_eval.append(sequence)
            labels_eval.append(label)
        else:
            sequences.append(sequence)
            labels.append(label)

sequences = np.array(sequences)
labels = np.array(labels)
sequences_eval = np.array(sequences_eval)
labels_eval = np.array(labels_eval)

# Convert to tensors
X_train = torch.tensor(sequences, dtype=torch.float)
y_train = torch.tensor(labels, dtype=torch.long)
X_test = torch.tensor(sequences_eval, dtype=torch.float)
y_test = torch.tensor(labels_eval, dtype=torch.long)

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


# Count the occurrences of each label
label_counts = Counter(labels)
none_freq = label_counts[0]/len(sequences)*100
left_freq = label_counts[1]/len(sequences)*100
right_freq = label_counts[2]/len(sequences)*100

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Define a weighted loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Train model
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float("Inf") 
epochs_no_improve = 0
n_epochs_stop = 100  # Stop training if validation loss does not improve after 100 epochs

# Initialize variables
best_accuracy = 0.0
best_epoch = 0
epochs_since_improvement = 0
n_stop_epochs = 10  # Number of epochs to stop training if the accuracy does not improve
best_model_state = None  # To store the state of the best model


# 1st approach
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
        outputs_val = model(X_test)
        val_loss = criterion(outputs_val, y_test)

    # Calculate predictions
    _, predicted = torch.max(outputs_val, 1)

    # Convert tensors to numpy arrays
    y_test_np = y_test.cpu().numpy()
    predicted_np = predicted.cpu().numpy()


    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
        print(f'Epoch {epoch+1} - Validation loss decreased ({val_loss:.6f}')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == n_epochs_stop:
            print(f'Optimal fitting reached. Stopping! (Epoch: {epoch+1})')
            break

model.load_state_dict(best_model_state)


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


# Print the total number of sequences and entries
print(f"Total number of sequences: {len(sequences)}")
print(f"Total number of entries: {len(training_data)}")
print(f"Number of 'none' sequences: {label_counts[0]}")
print(f"Number of 'left' sequences: {label_counts[1]}")
print(f"Number of 'right' sequences: {label_counts[2]}")
print(f"Percentage None: {round(none_freq, 2)}%")
print(f"Percentage Left: {round(left_freq, 2)}%")
print(f"Percentage Right: {round(right_freq, 2)}%")

print(f'Detect rate None   : {none_accuracy*100}%')
print(f'Detect rate Left   : {left_accuracy*100}%')
print(f'Detect rate Right  : {right_accuracy*100}%')
print(f'Detect rate Overall: {overall_accuracy*100}%')


# After training, save your model parameters
torch.save(model.state_dict(), 'model.pth')

# Load the trained model
model.load_state_dict(torch.load('model.pth'))

now = datetime.now()
formatted_now = now.strftime("%Y%m%d%H%M%S")
onnx_path = f"Assets/MLModels/model_{formatted_now}.onnx"

torch.onnx.export(model,               # model being run
                  X_train[:1],         # model input (or a tuple for multiple inputs)
                  onnx_path,           # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,    # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['sequence'],   # the model's input names
                  output_names = ['output']) # the model's output names