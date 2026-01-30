# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 01:12:57 2025

Raw

PCA + XG Boost 

@author: invitados fc
"""

import datetime
start_time = datetime.datetime.now()
print("\nStart time:", start_time)

# --- Libraries ---
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             auc, roc_auc_score, balanced_accuracy_score)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# --- Paths and Names ---
Data_Folder = r'C:\Users\invitados fc\Documents\Files JCTG\2025 Raman HLB\Base de Datos'
Results_Folder = r'C:\Users\invitados fc\Documents\Files JCTG\2025 Raman HLB\Results ML'

Model_Name = 'PCA_XGBoost_Raw'
Conf_Mat_Name = f'{Model_Name} Confusion Matrix.jpg'
Roc_Curve_Name = f'{Model_Name} ROC Curve.jpg'
Results_Name = f'{Model_Name} Results.csv'

# --- Plot configuration ---
def configure_plots():
    plt.rcParams['figure.dpi'] = 900
    plt.rcParams['font.size'] = 24
    plt.rcParams['lines.linewidth'] = 7
configure_plots()

# --- Load Data ---
Control_Class = pd.read_csv(os.path.join(Data_Folder, 'Raw Healthy.csv'), header=0, index_col=0)
HLB_Asymptomatic_Class = pd.read_csv(os.path.join(Data_Folder, 'Raw Asymptomatic HLB.csv'), header=0, index_col=0)
HLB_Symptomatic_Class = pd.read_csv(os.path.join(Data_Folder, 'Raw Symptomatic HLB.csv'), header=0, index_col=0)


n_control = Control_Class.shape[0]
n_asymptomatic = HLB_Asymptomatic_Class.shape[0]
n_symptomatic = HLB_Symptomatic_Class.shape[0]

print(f"\nData dimensions:\nControl: {n_control}\nAsymptomatic: {n_asymptomatic}\nSymptomatic: {n_symptomatic}")

# --- Prepare Arrays and Labels ---
X = np.concatenate((Control_Class.values, HLB_Asymptomatic_Class.values, HLB_Symptomatic_Class.values), axis=0)
y = np.concatenate((
    np.zeros(n_control),
    np.ones(n_asymptomatic),
    np.full(n_symptomatic, 2)), axis=0)

# Remove rows with NaN values
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
y = y[mask]

print(f"\nData dimensions after removing NaN values:\nControl: {sum(y==0)}\nAsymptomatic: {sum(y==1)}\nSymptomatic: {sum(y==2)}")

# --- Apply StandardScaler ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Apply PCA ---
pca = PCA(n_components=0.95, random_state=42)  # retain 95% of the variance
X_pca = pca.fit_transform(X_scaled)

print(f"Number of components employed: {pca.n_components_}")
total_explained_variance = np.sum(pca.explained_variance_ratio_) * 100
print(f"Total explained variance ratio: {total_explained_variance:.2f}%")




# --- Split data into training (60%), validation (20%), and test (20%) ---
X_train, X_remaining, y_train, y_remaining = train_test_split(X_pca, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, stratify=y_remaining, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# --- Define hyperparameter grid ---
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators': [50, 100, 200, 500],
    'gamma': [0, 0.1, 0.5, 1],
    'subsample': [0.5, 0.8, 1],
    'colsample_bytree': [0.5, 0.8, 1],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0, 0.1, 0.5, 1]
}

# --- Define scoring metrics ---
scoring = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro'
}

# --- Define stratified k-fold cross-validation object ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Define XGBoost classifier ---
xgb_class = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)

# --- Define grid search object ---
grid_search_xgb = GridSearchCV(xgb_class, param_grid, cv=skf, scoring=scoring, refit='accuracy')

# --- Perform grid search ---
grid_search_xgb.fit(X_train, y_train)

# --- Get best model and its performance ---
best_model_xgb = grid_search_xgb.best_estimator_
best_score_xgb = grid_search_xgb.best_score_

print(f"\nBest Parameters: {grid_search_xgb.best_params_}")
print(f"Best Accuracy: {best_score_xgb:.4f}")

# --- Evaluate best model on validation set ---
y_pred_val_xgb = best_model_xgb.predict(X_val)
y_score_val_xgb = best_model_xgb.predict_proba(X_val)

cm_val_xgb = confusion_matrix(y_val, y_pred_val_xgb)
print(f"\nValidation Confusion Matrix:\n{cm_val_xgb}")

# --- Evaluate best model on test set ---
y_pred_test_xgb = best_model_xgb.predict(X_test)
y_score_test_xgb = best_model_xgb.predict_proba(X_test)

cm_test_xgb = confusion_matrix(y_test, y_pred_test_xgb)
print(f"\nTest Confusion Matrix:\n{cm_test_xgb}")

# --- Plot Confusion Matrix ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_test_xgb, annot=True, fmt='d', cmap='Blues',
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
legend_ax.text(0, 0.7, '0: Control', fontsize=20)
legend_ax.text(0, 0.5, '1: Asymptomatic', fontsize=20)
legend_ax.text(0, 0.3, '2: Symptomatic', fontsize=20)

plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.savefig(os.path.join(Results_Folder, Conf_Mat_Name), bbox_inches='tight')
plt.show()


# --- Extended Metrics ---
TP=np.diag(cm_xgb);
FP=cm_xgb.sum(0)-TP; 
FN=cm_xgb.sum(1)-TP; 
TN=cm_xgb.sum()- (FP+FN+TP)
specificity=TN/(TN+FP); 
sensitivity=TP/(TP+FN);
bal_acc=(specificity+sensitivity)/2
auc_macro=roc_auc_score(y,y_score_xgb,multi_class='ovr',average='macro')
auc_micro=roc_auc_score(y,y_score_xgb,multi_class='ovr',average='micro')
report=classification_report(y,y_pred_xgb,target_names=['Control','Asymptomatic','Symptomatic'],output_dict=True)
perf=[]
for i,name in enumerate(['Control','Asymptomatic','Symptomatic']):
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
             'Balanced Acc':balanced_accuracy_score(y,y_pred_xgb),
             'AUC Macro':auc_macro,
             'AUC Micro':auc_micro})
pd.DataFrame(perf).to_csv(os.path.join(Results_Folder,Results_Name),index=False)

# --- ROC Curves ---
y_true_bin = label_binarize(y, classes=[0,1,2])
fpr,tpr,roc_auc={}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:,i], y_score_xgb[:,i])
    roc_auc[i]=auc(fpr[i],tpr[i])
fpr["micro"],tpr["micro"],_=roc_curve(y_true_bin.ravel(),y_score_xgb.ravel())
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
class_names = ['Control', 'Asymptomatic', 'Symptomatic']

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

end_time=datetime.datetime.now()
print("\nStart:",start_time,
      "\nEnd:",end_time,
      "\nElapsed:",end_time-start_time)

print(f"\nResults saved to {os.path.join(Results_Folder, Results_Name)}")


