# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 21:24:15 2025

@author: invitados fc
"""

import datetime
start_time = datetime.datetime.now()
print("\nStart time:", start_time)

# --- Libraries ---
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             auc, roc_auc_score, balanced_accuracy_score)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt

# --- Paths and Names ---
Data_Folder = r'C:\Users\invitados fc\Documents\Files JCTG\2025 Raman HLB\Base de Datos'
Image_Name = 'Raw_Spectrum.jpg'
os.chdir(Data_Folder)

# --- Plot configuration ---
def configure_plots():
    plt.rcParams['figure.dpi'] = 900
    plt.rcParams['font.size'] = 24
    plt.rcParams['lines.linewidth'] = 7
configure_plots()

# --- Load Data ---
Control_Class = pd.read_csv(os.path.join(Data_Folder, 'Raw Healthy.csv'), header=0, index_col=0).dropna()
HLB_Asymptomatic_Class = pd.read_csv(os.path.join(Data_Folder, 'Raw Asymptomatic HLB.csv'), header=0, index_col=0).dropna()
HLB_Symptomatic_Class = pd.read_csv(os.path.join(Data_Folder, 'Raw Symptomatic HLB.csv'), header=0, index_col=0).dropna()

n_control = Control_Class.shape[0]
n_asymptomatic = HLB_Asymptomatic_Class.shape[0]
n_symptomatic = HLB_Symptomatic_Class.shape[0]

print(f"\nData dimensions:\nControl: {n_control}\nAsymptomatic: {n_asymptomatic}\nSymptomatic: {n_symptomatic}")

colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00']  # Colores para daltónicos
class_names = ['Control', 'Asymptomatic', 'Symptomatic']

# Plot configuration
plt.figure(figsize=(15, 8))

# Calculate mean and std for each class
control_mean = Control_Class.mean(axis=0)
asymptomatic_mean = HLB_Asymptomatic_Class.mean(axis=0)
symptomatic_mean = HLB_Symptomatic_Class.mean(axis=0)

control_std = Control_Class.std(axis=0)
asymptomatic_std = HLB_Asymptomatic_Class.std(axis=0)
symptomatic_std = HLB_Symptomatic_Class.std(axis=0)

wavelengths = np.array(Control_Class.columns, dtype=float)

# Plot mean and std
plt.plot(wavelengths, control_mean, label='Control', color=colors[0])
plt.fill_between(wavelengths, control_mean - control_std, control_mean + control_std, alpha=0.2, color=colors[0])

plt.plot(wavelengths, asymptomatic_mean, label='Asymptomatic', color=colors[1])
plt.fill_between(wavelengths, asymptomatic_mean - asymptomatic_std, asymptomatic_mean + asymptomatic_std, alpha=0.2, color=colors[1])

plt.plot(wavelengths, symptomatic_mean, label='Symptomatic', color=colors[2])
plt.fill_between(wavelengths, symptomatic_mean - symptomatic_std, symptomatic_mean + symptomatic_std, alpha=0.2, color=colors[2])

plt.xlabel('Wavelength (cm^-1)')
plt.ylabel('Intensity')
plt.title('Raman Spectra')
plt.xticks(np.linspace(wavelengths.min(), wavelengths.max(), 5))
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(Data_Folder, Image_Name), bbox_inches='tight')
plt.show()

end_time = datetime.datetime.now()
print("\nEnd time:", end_time)
print("Execution time:", end_time - start_time)