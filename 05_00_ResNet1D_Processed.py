# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 18:33:33 2025

@author: invitados fc
"""



import datetime
start_time = datetime.datetime.now()
print("\nStart time:", start_time)

# --- Libraries ---
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             auc, roc_auc_score, balanced_accuracy_score)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt

# --- Paths and Names ---
Data_Folder = r'C:\Users\invitados fc\Documents\Files JCTG\2025 Raman HLB\Base de Datos'
Results_Folder = r'C:\Users\invitados fc\Documents\Files JCTG\2025 Raman HLB\Results Folder'

Model_Name = '05_00_ResNet1D_Processed'
Conf_Mat_Name = f'{Model_Name} Confusion Matrix.jpg'
Roc_Curve_Name = f'{Model_Name} ROC Curve.jpg'
Results_Name = f'{Model_Name} Results.csv'
Loss_Acc = f'{Model_Name} Loss and Accuracy.jpg'
Best_Model_Path = os.path.join(Results_Folder, f'{Model_Name}_best_model.pth')
Best_Model_Path_model = os.path.join(Results_Folder, f'{Model_Name}_best_all_model.pth')



# --- Plot configuration ---
def configure_plots():
    plt.rcParams['figure.dpi'] = 900
    plt.rcParams['font.size'] = 24
    plt.rcParams['lines.linewidth'] = 3
configure_plots()

# --- Load Data ---
Healthy_Class = pd.read_csv(os.path.join(Data_Folder, 'Processed Healthy.csv'), header=0, index_col=0).dropna()
HLB_Asymptomatic_Class = pd.read_csv(os.path.join(Data_Folder, 'Processed Asymptomatic HLB.csv'), header=0, index_col=0).dropna()
HLB_Symptomatic_Class = pd.read_csv(os.path.join(Data_Folder, 'Processed Symptomatic HLB.csv'), header=0, index_col=0).dropna()

n_healthy, n_asymptomatic, n_symptomatic = Healthy_Class.shape[0], HLB_Asymptomatic_Class.shape[0], HLB_Symptomatic_Class.shape[0]
print(f"\nData dimensions:\nHealthy: {n_healthy}\nAsymptomatic: {n_asymptomatic}\nSymptomatic: {n_symptomatic}")

# --- Prepare Arrays and Labels ---
X = np.concatenate((Healthy_Class.values, HLB_Asymptomatic_Class.values, HLB_Symptomatic_Class.values), axis=0)
y = np.concatenate((
    np.zeros(n_healthy),
    np.ones(n_asymptomatic),
    np.full(n_symptomatic, 2)), axis=0)

# --- Split Data ---
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

print(f"\nDataset dimensions:\nX_train: {X_train.shape}, y_train: {y_train.shape}\nX_val: {X_val.shape}\nX_test: {X_test.shape}")

# --- Class Weights ---
train_counts = np.unique(y_train, return_counts=True)[1]
class_counts = torch.tensor(train_counts, dtype=torch.float32)
total_samples = class_counts.sum()
n_classes = len(class_counts)
class_weights = total_samples / (n_classes * class_counts)
print("\nClass Weights:", class_weights.numpy())

# --- Dataset and DataLoaders ---
class RamanDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(RamanDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(RamanDataset(X_val, y_val), batch_size=32, shuffle=False)
test_loader = DataLoader(RamanDataset(X_test, y_test), batch_size=32, shuffle=False)

# --- ResNet1D Model ---
# --- ResNet1D Model ---
class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class Bottleneck1D(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class SimpleResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=3):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion))
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks): layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.fc(self.avgpool(x).view(x.size(0), -1))
        return x

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleResNet1D(Bottleneck1D, [3, 8, 36, 3], num_classes=3).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

# --- Train ---
def evaluate(model, loader):
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            loss += criterion(out, yb).item()
            correct += (out.argmax(1)==yb).sum().item()
    return loss/len(loader), correct/len(loader.dataset)

EPOCHS, best_val, patience, trigger = 100, float('inf'), 10, 0
history = {'train_loss':[], 'val_loss':[], 'val_accuracy':[]}
print("\n--- Starting Training ---")
for e in range(EPOCHS):
    model.train(); total=0
    for Xb,yb in train_loader:
        Xb,yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad(); out=model(Xb)
        loss=criterion(out,yb); loss.backward(); optimizer.step()
        total+=loss.item()
    val_loss,val_acc=evaluate(model,val_loader)
    scheduler.step(val_loss)
    print(f"Epoch {e+1}: Train {total/len(train_loader):.4f}, Val {val_loss:.4f}, Acc {val_acc:.4f}")
    history['train_loss'].append(total/len(train_loader))
    history['val_loss'].append(val_loss); history['val_accuracy'].append(val_acc)
    if val_loss<best_val:
        best_val=val_loss; trigger=0
        torch.save(model.state_dict(), Best_Model_Path)
        torch.save(model, Best_Model_Path_model)
    else:
        trigger+=1
        if trigger>=patience:
            print("Early stopping.")
            break

# --- Evaluation ---
model.load_state_dict(torch.load(Best_Model_Path))
test_loss,test_acc=evaluate(model,test_loader)
print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%")

# --- Predictions ---

model.eval();

y_true=[]; 
y_pred=[]; 
y_score=[]

with torch.no_grad():
    for Xb,yb in test_loader:
        Xb,yb=Xb.to(device), yb.to(device)
        out=model(Xb)
        probs=torch.softmax(out,dim=1)
        y_pred.extend(out.argmax(1).cpu().numpy())
        y_true.extend(yb.cpu().numpy())
        y_score.extend(probs.cpu().numpy())

y_true=np.array(y_true); 
y_pred=np.array(y_pred); 
y_score=np.array(y_score)
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

plt.savefig(os.path.join(Results_Folder, Roc_Curve_Name), 
            bbox_inches='tight')
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
plt.savefig(os.path.join(Results_Folder, Loss_Acc), 
            bbox_inches='tight')
plt.show()

end_time=datetime.datetime.now()
print("\nStart:",start_time,
      "\nEnd:",end_time,
      "\nElapsed:",end_time-start_time)

print(f"\nResults saved to {os.path.join(Results_Folder, Results_Name)}")
