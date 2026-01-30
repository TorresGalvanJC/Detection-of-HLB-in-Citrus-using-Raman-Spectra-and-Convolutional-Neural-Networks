# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 01:05:10 2025


Raman Processed

PCA + SVM

Sí se aplica PCA


@author: invitados fc
"""


import datetime
start_time = datetime.datetime.now()
print("\nStart time:", start_time)

# --- Libraries ---
import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             auc, roc_auc_score, balanced_accuracy_score)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA # NEW: Import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# --- Paths and Names ---
# NOTE: Replace with your actual paths
Data_Folder = r'C:\Users\invitados fc\Documents\Files JCTG\2025 Raman HLB\Base de Datos'
Results_Folder = r'C:\Users\invitados fc\Documents\Files JCTG\2025 Raman HLB\Results ML'

Model__Name = 'PCA_SVM_Processed_' # Updated name to reflect PCA
Conf_Mat_Name = f'{Model__Name} Confusion Matrix.jpg'
Roc_Curve_Name = f'{Model__Name} ROC Curve.jpg'
Results_Name = f'{Model__Name} Results.csv'

# --- Plot configuration ---
def configure_plots():
    plt.rcParams['figure.dpi'] = 900
    plt.rcParams['font.size'] = 24
    plt.rcParams['lines.linewidth'] = 3
configure_plots()

# --- Load Data ---
Control_Class = pd.read_csv(os.path.join(Data_Folder, 'Processed Healthy.csv'), header=0, index_col=0)
HLB_Asymptomatic_Class = pd.read_csv(os.path.join(Data_Folder, 'Processed Asymptomatic HLB.csv'), header=0, index_col=0)
HLB_Symptomatic_Class = pd.read_csv(os.path.join(Data_Folder, 'Processed Symptomatic HLB.csv'), header=0, index_col=0)


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

# ==============================================================================
# --- Data Split (60% Train, 20% Validation, 20% Test) ---
# ==============================================================================

# 1. Split into Test (20%) and Train+Val (80%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 2. Split Train+Val (80%) into Train (60% of total) and Validation (20% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

print("\n--- Data Split Summary ---")
print(f"Total samples: {len(y)}")
print(f"Training set (60%): {len(y_train)} samples")
print(f"Validation set (20%): {len(y_val)} samples")
print(f"Test set (20%): {len(y_test)} samples")
print("--------------------------")

# ==============================================================================
# --- NEW: Apply PCA for Dimensionality Reduction ---
# We will select the number of components to retain 95% of the variance.
# PCA must be fit ONLY on X_train_val to prevent leakage into the test set.
# ==============================================================================

# Initialize PCA and fit on the combined Training + Validation set
# We use X_train_val for fitting because GridSearchCV will cross-validate on this set.
pca = PCA(n_components=0.95, random_state=42)
pca.fit(X_train_val)

# Transform the datasets
X_train_val_pca = pca.transform(X_train_val)
X_test_pca = pca.transform(X_test)

n_components = pca.n_components_
print(f"\nPCA reduced the dimensionality to {n_components} components, retaining 95% of variance.")
print(f"Original features: {X.shape[1]}")
print(f"PCA features: {n_components}")


print(f"Number of components employed: {pca.n_components_}")
total_explained_variance = np.sum(pca.explained_variance_ratio_) * 100
print(f"Total explained variance ratio: {total_explained_variance:.2f}%")


# The features for GridSearchCV fitting are now the PCA components
X_fit = X_train_val_pca
y_fit = y_train_val


# --- Define hyperparameter grid ---
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# --- Define scoring metrics ---
scoring = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro'
}

# --- Define stratified k-fold cross-validation object ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Define SVM classifier ---
svm = SVC(probability=True, random_state=42)

# --- Define grid search object ---
# Grid search is performed on the PCA-transformed Training+Validation set (X_fit)
grid_search = GridSearchCV(svm, param_grid, cv=skf, scoring=scoring, refit='accuracy')

# --- Perform grid search ---
# Fitting only on the combined Training and Validation data (PCA components)
grid_search.fit(X_fit, y_fit)

# --- Get best model and its performance ---
best_model = grid_search.best_estimator_
best_cv_score = grid_search.best_score_ 

print(f"\nBest Parameters (from 5-Fold CV on 80% data with PCA): {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {best_cv_score:.4f}")

# ==============================================================================
# --- Evaluate best model on the independent Test Set (20%) ---
# The evaluation data is now the PCA-transformed test set (X_test_pca)
# ==============================================================================
print("\n--- Model Evaluation on Independent 20% Test Set (PCA) ---")
X_final_eval = X_test_pca
y_true_final = y_test

# Make predictions on the unseen Test Set (PCA components)
y_pred = best_model.predict(X_final_eval)
y_score = best_model.predict_proba(X_final_eval)

cm = confusion_matrix(y_true_final, y_pred)
print(f"\nTest Set Confusion Matrix:\n{cm}")

# --- Plot Confusion Matrix (Test Set) ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[0, 1, 2],
            yticklabels=[0, 1, 2],
            ax=ax)
ax.set_title('Test Set Confusion Matrix (PCA)', pad=20)
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

# --- Extended Metrics (Test Set) ---
TP=np.diag(cm);
FP=cm.sum(0)-TP; 
FN=cm.sum(1)-TP; 
TN=cm.sum()- (FP+FN+TP)
specificity=TN/(TN+FP); 
sensitivity=TP/(TP+FN);
bal_acc=(specificity+sensitivity)/2
auc_macro=roc_auc_score(y_true_final, y_score, multi_class='ovr', average='macro')
auc_micro=roc_auc_score(y_true_final, y_score, multi_class='ovr', average='micro')
report=classification_report(y_true_final, y_pred, target_names=['Control','Asymptomatic','Symptomatic'], output_dict=True)
perf=[]
for i,name in enumerate(['Control','Asymptomatic','Symptomatic']):
    perf.append({'Class':name,
                 'Precision':report[name]['precision'],
                 'Recall':sensitivity[i],
                 'Specificity':specificity[i],
                 'Balanced Acc':bal_acc[i],
                 'F1':report[name]['f1-score']})
perf.append({'Class':'Global Avg (Test Set PCA)',
             'Precision':report['macro avg']['precision'],
             'Recall':report['macro avg']['recall'],
             'F1':report['macro avg']['f1-score'],
             'Balanced Acc':balanced_accuracy_score(y_true_final, y_pred),
             'AUC Macro':auc_macro,
             'AUC Micro':auc_micro})
pd.DataFrame(perf).to_csv(os.path.join(Results_Folder,Results_Name),index=False)

# --- ROC Curves (Test Set) ---
y_true_bin = label_binarize(y_true_final, classes=[0,1,2])
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
class_names = ['Control', 'Asymptomatic', 'Symptomatic']

for i, c in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=c, label=f'{class_names[i]} (AUC={roc_auc[i]:.3f})')

plt.plot(fpr["micro"], tpr["micro"], '--', color='deeppink', label=f'Micro (AUC={roc_auc["micro"]:.3f})')
plt.plot(fpr["macro"], tpr["macro"], ':', color='navy', label=f'Macro (AUC={roc_auc["macro"]:.3f})')
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('Test Set ROC Curves (PCA)', pad=20) 

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