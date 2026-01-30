# Detection-of-HLB-in-Citrus-using-Raman-Spectra-and-Convolutional-Neural-Networks
Evaluation of Raman spectra of orange trees with symptoms of HLB, asymptomatic HLB and controls. Different machine learning techniques and approaches of ResNet to be classified in each category employing the raw data and processed to eliminate the fluorescence.

Objective: To evaluate the performance of 1D Convolutional Neural Networks (1D-CNN), specifically ResNet architectures, for HLB detection using Raman spectroscopy by comparing raw versus pre-processed spectral data. 

Methods: 1D ResNet-50 and ResNet-152 models were trained from scratch and via transfer learning to classify Raman spectra from control, symptomatic, and asymptomatic citrus leaves. 

Results: The ResNet-152 architecture trained from scratch on raw spectra outperformed all other models. Crucially, raw data, including biological fluorescence, provided superior diagnostic markers compared to pre-processed spectra, significantly improving the identification of asymptomatic samples. 

Conclusions: Deep 1D residual networks trained on raw Raman signals provide a high-precision, non-invasive tool for early HLB detection. This approach eliminates the need for complex signal pre-processing, facilitating the development of real-time, cost-effective diagnostic platforms for sustainable citrus management.

Please cite this codes as... Paper is under review
