# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 02:16:50 2025

@author: invitados fc
"""

import os
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             auc, roc_auc_score, balanced_accuracy_score)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

start_time = datetime.datetime.now()
print("\nStart time:", start_time)

# Define paths and names
Data_Folder = r'C:\Users\invitados fc\Documents\Files JCTG\2025 Raman HLB\Base de Datos'
Results_Folder = r'C:\Users\invitados fc\Documents\Files JCTG\2025 Raman HLB\Results ML'

Model_Name = 'kNN_PCA_Processed'
Conf_Mat_Name = f'{Model_Name} Confusion Matrix.jpg'
Roc_Curve_Name = f'{Model_Name} ROC Curve.jpg'
Results_Name = f'{Model_Name} Results.csv'

# Create Results_Folder if it doesn't exist
if not os.path.exists(Results_Folder):
    os.makedirs(Results_Folder)

# Load data
Control_Class = pd.read_csv(os.path.join(Data_Folder, 'Processed Healthy.csv'), header=0, index_col=0)
HLB_Asymptomatic_Class = pd.read_csv(os.path.join(Data_Folder, 'Processed Asymptomatic HLB.csv'), header=0, index_col=0)
HLB_Symptomatic_Class = pd.read_csv(os.path.join(Data_Folder, 'Processed Symptomatic HLB.csv'), header=0, index_col=0)

# Prepare arrays and labels
X = np.vstack((Control_Class.values, HLB_Asymptomatic_Class.values, HLB_Symptomatic_Class.values))
y = np.concatenate((np.zeros(Control_Class.shape[0]), np.ones(HLB_Asymptomatic_Class.shape[0]), np.full(HLB_Symptomatic_Class.shape[0], 2)))

# Remove rows with NaN values
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
y = y[mask]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PCA Transformation
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"Number of components employed: {pca.n_components_}")
total_explained_variance = np.sum(pca.explained_variance_ratio_) * 100
print(f"Total explained variance ratio: {total_explained_variance:.2f}%")


# Define hyperparameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30, 40]
}

# Define scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro'
}

# Define stratified k-fold cross-validation object
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define kNN classifier
knn_class = KNeighborsClassifier()

# Define grid search object
grid_search_knn = GridSearchCV(knn_class, param_grid, cv=skf, scoring=scoring, refit='accuracy')

# Perform grid search
grid_search_knn.fit(X_train_pca, y_train)

# Get best model and its performance
best_model_knn = grid_search_knn.best_estimator_
best_score_knn = grid_search_knn.best_score_

print(f"\nBest Parameters: {grid_search_knn.best_params_}")
print(f"Best Accuracy: {best_score_knn:.4f}")

# Evaluate best model
y_pred_knn = best_model_knn.predict(X_test_pca)
y_score_knn = best_model_knn.predict_proba(X_test_pca)

cm_knn = confusion_matrix(y_test, y_pred_knn)
print(f"\nConfusion Matrix:\n{cm_knn}")

# Plot configuration
def configure_plots():
    plt.rcParams['figure.dpi'] = 900
    plt.rcParams['font.size'] = 24
    plt.rcParams['lines.linewidth'] = 7
configure_plots()

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues',
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

# Extended Metrics
TP=np.diag(cm_knn);
FP=cm_knn.sum(0)-TP; 
FN=cm_knn.sum(1)-TP; 
TN=cm_knn.sum()- (FP+FN+TP)
specificity=TN/(TN+FP); 
sensitivity=TP/(TP+FN);
bal_acc=(specificity+sensitivity)/2
auc_macro=roc_auc_score(y_test,y_score_knn,multi_class='ovr',average='macro')
auc_micro=roc_auc_score(y_test,y_score_knn,multi_class='ovr',average='micro')
report=classification_report(y_test,y_pred_knn,target_names=['Control','Asymptomatic','Symptomatic'],output_dict=True)
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
             'Balanced Acc':balanced_accuracy_score(y_test,y_pred_knn),
             'AUC Macro':auc_macro,
             'AUC Micro':auc_micro})
pd.DataFrame(perf).to_csv(os.path.join(Results_Folder,Results_Name),index=False)

# ROC Curves
y_true_bin = label_binarize(y_test, classes=[0,1,2])
fpr,tpr,roc_auc={}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:,i], y_score_knn[:,i])
    roc_auc[i]=auc(fpr[i],tpr[i])
fpr["micro"],tpr["micro"],_=roc_curve(y_true_bin.ravel(),y_score_knn.ravel())
roc_auc["micro"]=auc(fpr["micro"],tpr["micro"])
all_fpr=np.unique(np.concatenate([fpr[i] for i in range(3)]))
mean_tpr=np.zeros_like(all_fpr)
for i in range(3): mean_tpr+=np.interp(all_fpr,fpr[i],tpr[i]) 
mean_tpr/=3; 
fpr["macro"]=all_fpr
tpr["macro"]=mean_tpr
roc_auc["macro"]=auc(fpr["macro"],tpr["macro"])

plt.figure(figsize=(10, 10))  
colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00']  
class_names = ['Control', 'Asymptomatic', 'Symptomatic']

for i, c in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=c, label=f'{class_names[i]} (AUC={roc_auc[i]:.3f})')

plt.plot(fpr["micro"], tpr["micro"], '--', color='deeppink', label=f'Micro (AUC={roc_auc["micro"]:.3f})')
plt.plot(fpr["macro"], tpr["macro"], ':', color='navy', label=f'Macro (AUC={roc_auc["macro"]:.3f})')
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curves', pad=20)  

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