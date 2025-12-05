# MoE-Net: Genetic Algorithm–Optimized Mixture of Experts for Nasal Bone Fracture Detection

This repository contains the official implementation of MoE-Net, a deep ensemble framework optimized by a Genetic Algorithm for automated detection of nasal bone fractures in lateral-view radiographs.
The method integrates three fine-tuned CNN experts with GA-based weight optimization and includes a domain-informed preprocessing pipeline and Grad-CAM interpretability.

 ## Dataset (Example Images)
<p align="center"> <img src="Images/Dataset.png" width="80%"> </p>

The dataset contains lateral-view radiographs labeled as normal or fractured, with diverse fracture patterns such as step-off deformity, angulation, and cortical discontinuity.

## Preprocessing Pipeline
<p align="center"> <img src="Images/Preprocessing.png" width="80%"> </p>

The preprocessing workflow includes:

ROI extraction to isolate the nasal bone region

CLAHE for improved contrast

Sharpening to enhance anatomical edges

Gaussian noise to simulate real radiographic variability

Controlled augmentation to increase robustness

## MoE-Net Architecture
<p align="center"> <img src="Images/Ensemble Model.png" width="85%"> </p>

MoE-Net combines three expert CNN models—InceptionResNetV2, DenseNet121, and Xception—selected based on diagnostic performance.
A Genetic Algorithm optimizes expert contribution weights, allowing the ensemble to balance sensitivity and specificity effectively.

## Grad-CAM Interpretability
<p align="center"> <img src="Images/Grad-CAM.png" width="80%"> </p>

Grad-CAM visualizations highlight the regions most influential in the model’s decision-making, increasing interpretability and supporting potential clinical adoption.

## Acknowledgements

I gratefully acknowledge the contributions and support of:

- Prof. Seyed Abolghasem Mirroshandel — supervision and scientific guidance  
- Dr. Tahereh Mortezaei — clinical expertise and dataset supervision  
- Dr. Zahra Dalili Kajan — radiology annotations and clinical validation  

Their expertise and collaboration were essential to the success of this project.
