# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 12:37:13 2025

@author: invitados fc
"""

import datetime
start_time = datetime.datetime.now()
print("\nStart time:", start_time)

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, auc, precision_recall_fscore_support, 
    accuracy_score, balanced_accuracy_score)
import seaborn as sns
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Variables configuration
# NOTE: Ensure these paths exist on your system before running.
Data_Folder = r'C:\Users\invitados fc\Documents\Files JCTG\2025 Raman HLB\Base de Datos'
Results_Folder = r'C:\Users\invitados fc\Documents\Files JCTG\2025 Raman HLB\Results Folder'

Model_Name = '04_ResNet1D_Optimized' # Updated model name for the optimized version
Conf_Mat_Name = f'{Model_Name} Confusion Matrix.jpg'
Roc_Curve_Name = f'{Model_Name} ROC Curve.jpg'
Results_Name = f'{Model_Name} Results.csv'
Loss_vs_Epoch = f'{Model_Name} Loss vs Epoch.jpg'
Acc_Vs_Epoch = f'{Model_Name} Accuracy vs Epoch.jpg'
Loss_Acc = f'{Model_Name} Loss and Accuracy.jpg'
Best_Model_Path = os.path.join(Results_Folder, f'{Model_Name}_best_model.pth')
Best_Model_Path_model = os.path.join(Results_Folder, f'{Model_Name}_best_all_model.pth')


# Function to configure plots
def configure_plots():
    plt.rcParams['figure.dpi'] = 900
    plt.rcParams['font.size'] = 24
    plt.rcParams['lines.linewidth'] = 3

configure_plots()

# --- Data Loading and Preprocessing ---

# Load data
Healthy_Class = pd.read_csv(os.path.join(Data_Folder, 'Processed Healthy.csv'), header=0, index_col=0)
HLB_Asymptomatic_Class = pd.read_csv(os.path.join(Data_Folder, 'Processed Asymptomatic HLB.csv'), header=0, index_col=0)
HLB_Symptomatic_Class = pd.read_csv(os.path.join(Data_Folder, 'Processed Symptomatic HLB.csv'), header=0, index_col=0)

# Remove rows with NaN values
Healthy_Class = Healthy_Class.dropna()
HLB_Asymptomatic_Class = HLB_Asymptomatic_Class.dropna()
HLB_Symptomatic_Class = HLB_Symptomatic_Class.dropna()

## Check Number of elements for each class
n_healthy = Healthy_Class.shape[0]
n_asymptomatic = HLB_Asymptomatic_Class.shape[0]
n_symptomatic = HLB_Symptomatic_Class.shape[0]

# Check data dimensions
print("\n\n Data dimensions:")
print(f"Healthy_Class: {n_healthy}")
print(f"HLB_Asymptomatic_Class: {n_asymptomatic}")
print(f"HLB_Symptomatic_Class: {n_symptomatic}")

# Convert dataframes to numpy arrays
X_healthy = Healthy_Class.values
X_asymptomatic = HLB_Asymptomatic_Class.values
X_symptomatic = HLB_Symptomatic_Class.values

# Create labels
y_healthy = np.zeros(n_healthy)
y_asymptomatic = np.ones(n_asymptomatic)
y_symptomatic = np.full(n_symptomatic, 2)

# Concatenate arrays
X = np.concatenate((X_healthy, X_asymptomatic, X_symptomatic), axis=0)
y = np.concatenate((y_healthy, y_asymptomatic, y_symptomatic), axis=0)

# Split data into training, testing, and validation sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val) # 0.25 of 0.8 is 0.2

print("\n\n Dimensions of the datasets:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Verify the number of elements in each class
print("\n\n Cantidad de elementos en cada clase:")
train_counts = np.unique(y_train, return_counts=True)[1]
print("Entrenamiento:", train_counts)
print("Validación:", np.unique(y_val, return_counts=True)[1])
print("Prueba:", np.unique(y_test, return_counts=True)[1])


# --- Class Weights Calculation ---
# 1. Calculate the weight for each class on the training set
class_counts = torch.tensor(train_counts, dtype=torch.float32)
total_samples = class_counts.sum()
n_classes = len(class_counts)

# Weights inversely proportional to class frequencies
class_weights = total_samples / (n_classes * class_counts)

print("\nClass Weights (0: Healthy, 1: HLB Asymptomatic, 2: HLB Symptomatic):", class_weights.cpu().numpy())


# --- PyTorch Dataset and DataLoader Setup ---

# Create datasets and dataloaders
class RamanDataset(Dataset):
    def __init__(self, X, y):
        # Convert X to 2D numpy array if it's not already (for safety)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = RamanDataset(X_train, y_train)
val_dataset = RamanDataset(X_val, y_val)
test_dataset = RamanDataset(X_test, y_test)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --- Simplified ResNet1D Model Definition ---

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        # 1st convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 2nd convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out) # Final ReLU after addition

        return out

class SimpleResNet1D(nn.Module):
    # Layers configuration for ResNet-18 equivalent: [2, 2, 2, 2]
    def __init__(self, block, layers, num_classes=3):
        super(SimpleResNet1D, self).__init__()
        self.in_channels = 64
        
        # Initial layer
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet Stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Add channel dimension: (batch_size, sequence_length) -> (batch_size, 1, sequence_length)
        x = x.unsqueeze(1) 
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# --- Model Instantiation and Hyperparameter Setup ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the simplified model (ResNet-18 equivalent: [2, 2, 2, 2])
model = SimpleResNet1D(BasicBlock1D, [2, 2, 2, 2], num_classes=3).to(device)

# Move weights to the same device
weights = class_weights.to(device)

# Loss function with Class Weights
criterion = nn.CrossEntropyLoss(weight=weights)

# Optimizer with Learning Rate and L2 Regularization (weight_decay)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # 1e-4 is a good starting L2 regularization value

# Learning Rate Scheduler (Reduces LR by factor 0.5 if val_loss doesn't improve for 5 epochs)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


# --- Training and Evaluation Functions ---

def evaluate(model, device, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()

    accuracy = correct / len(loader.dataset)
    test_loss /= len(loader)
    return test_loss, accuracy


# --- Training Loop with Early Stopping ---

EPOCHS = 100
history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
best_val_loss = float('inf')
patience = 10 # Early stopping patience
trigger_times = 0

print("\n--- Starting Training ---")
for epoch in range(EPOCHS):
    # Training
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)
    history['train_loss'].append(train_loss)
    
    # Evaluation on Validation Set
    val_loss, val_accuracy = evaluate(model, device, val_loader)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_accuracy)

    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    # Learning Rate Scheduler step (based on validation loss)
    scheduler.step(val_loss)
    
    # Early Stopping Check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        # Save the best model weights
        torch.save(model.state_dict(), Best_Model_Path)
        torch.save(model, Best_Model_Path_model)
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f'Early stopping triggered after {patience} epochs without improvement.')
            break

print("\n--- Final Evaluation ---")
# Load the best model weights before final testing
model.load_state_dict(torch.load(Best_Model_Path))

# Evaluate model on test set
test_loss, test_accuracy = evaluate(model, device, test_loader)
print(f'Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy*100:.2f}%')


# --- Performance Metrics and Plotting ---

# Evaluate model on test set for predictions
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(target.cpu().numpy())
        model.eval()
y_prob = []
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        output = model(data)
        probs = torch.softmax(output, dim=1)
        y_prob.extend(probs.cpu().numpy())
y_prob = np.array(y_prob)

try:
    auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    auc_micro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='micro')
except Exception:
    auc_macro = np.nan
    auc_micro = np.nan

# Calculate confusion matrix
y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm=confusion_matrix(y_true,y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[0, 1, 2],
            yticklabels=[0, 1, 2],
            ax=ax)
ax.set_title('Confusion Matrix', pad=20)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')

# Adding a legend to explain the classes
legend_ax = plt.axes([0.8, 0.1, 0.15, 0.3])
legend_ax.axis('off')
legend_ax.text(0, 0.9, 'Class:', fontsize=20)
legend_ax.text(0, 0.7, '0: Healthy', fontsize=20)
legend_ax.text(0, 0.5, '1: Asymptomatic', fontsize=20)
legend_ax.text(0, 0.3, '2: Symptomatic', fontsize=20)

plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.savefig(os.path.join(Results_Folder, Conf_Mat_Name), bbox_inches='tight')
plt.show()

# --- Extended Metrics ---
TP=np.diag(cm);
FP=cm.sum(0)-TP; 
FN=cm.sum(1)-TP; 
TN=cm.sum()- (FP+FN+TP)
specificity=TN/(TN+FP); 
sensitivity=TP/(TP+FN);
bal_acc=(specificity+sensitivity)/2
auc_macro=roc_auc_score(y_true,y_score,multi_class='ovr',average='macro')
auc_micro=roc_auc_score(y_true,y_score,multi_class='ovr',average='micro')
report=classification_report(y_true,y_pred,target_names=['Healthy','Asymptomatic','Symptomatic'],output_dict=True)
perf=[]
for i,name in enumerate(['Healthy','Asymptomatic','Symptomatic']):
    perf.append({'Class':name,
                 'Precision':report[name]['precision'],
                 'Recall':sensitivity[i],
                 'Specificity':specificity[i],
                 'Balanced Acc':bal_acc[i],
                 'F1':report[name]['f1-score']})
perf.append({'Class':'Global Avg',
             'Precision':report['macro avg']['precision'],
             'Recall':report['macro avg']['recall'],
             'F1':report['macro avg']['f1-score'],
             'Balanced Acc':balanced_accuracy_score(y_true,y_pred),
             'AUC Macro':auc_macro,
             'AUC Micro':auc_micro})
pd.DataFrame(perf).to_csv(os.path.join(Results_Folder,Results_Name),index=False)

# --- ROC Curves ---

y_true_bin = label_binarize(y_true, classes=[0,1,2])
fpr,tpr,roc_auc={}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:,i], y_score[:,i])
    roc_auc[i]=auc(fpr[i],tpr[i])
fpr["micro"],tpr["micro"],_=roc_curve(y_true_bin.ravel(),y_score.ravel())
roc_auc["micro"]=auc(fpr["micro"],tpr["micro"])
all_fpr=np.unique(np.concatenate([fpr[i] for i in range(3)]))
mean_tpr=np.zeros_like(all_fpr)
for i in range(3): mean_tpr+=np.interp(all_fpr,fpr[i],tpr[i])
mean_tpr/=3; 
fpr["macro"]=all_fpr
tpr["macro"]=mean_tpr
roc_auc["macro"]=auc(fpr["macro"],tpr["macro"])

plt.figure(figsize=(10, 10))  # Gráfica cuadrada
colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00']  # Colores para daltónicos
class_names = ['Healthy', 'Asymptomatic', 'Symptomatic']

for i, c in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=c, label=f'{class_names[i]} (AUC={roc_auc[i]:.3f})')

plt.plot(fpr["micro"], tpr["micro"], '--', color='deeppink', label=f'Micro (AUC={roc_auc["micro"]:.3f})')
plt.plot(fpr["macro"], tpr["macro"], ':', color='navy', label=f'Macro (AUC={roc_auc["macro"]:.3f})')
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curves', pad=20)  # Separate title and graph

# Adjust the legend's position
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# Adjust the graph design
plt.tight_layout(rect=[0, 0, 0.75, 1])

plt.savefig(os.path.join(Results_Folder, Roc_Curve_Name), bbox_inches='tight')
plt.show()

# --- Loss & Accuracy curves ---
plt.figure(figsize=(11, 12))
plt.subplot(2, 1, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(Results_Folder, Loss_Acc), bbox_inches='tight')
plt.show()

end_time = datetime.datetime.now()
print("\n Start time:", start_time)
print("\n End time:", end_time)
print('\n Time spent : {}'.format(end_time - start_time))