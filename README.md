# Breast Ultrasound Image Segmentation

## Clinical Context

Breast cancer continues to be a major cause of mortality among women worldwide. Early detection significantly improves patient outcomes, and ultrasound imaging provides a non-invasive, radiation-free method for screening breast tissue abnormalities.

---

## Dataset Description

### Imaging Data

- **Modality**: B-mode ultrasound scans  
- **Resolution**: Originally 500×500 pixels (resized to 256×256 pixels for efficiency)  
- **Format**: PNG files  
- **Annotations**: Paired ground truth masks available  
- **Classes**:
  - **Normal**: Healthy breast tissue
  - **Benign**: Non-cancerous lesions
  - **Malignant**: Cancerous tumors

### Demographics

- **Cohort Size**: 600 female patients  
- **Age Range**: 25–75 years  
- **Collection Year**: 2018  

---

## Technical Utility

This dataset is structured to support **multi-class segmentation** tasks in medical imaging and enables:

- **Lesion Delineation**: Pixel-wise classification of tissue types  
- **Tumor Characterization**: Differentiation between benign and malignant tumors  
- **Deep Learning Integration**: Suited for training with architectures like Attention UNet  

---

## Limitations & Ethical Considerations

- **Temporal Relevance**: Dataset collected in 2018; applicability to current clinical settings requires validation  
- **Modality Exclusivity**: Focused solely on ultrasound; no mammography or MRI data included  
- **Clinical Disclaimer**: Intended for research purposes only, not validated for clinical decision-making  

---

## Model Selection Rationale

### Attention UNet

An **Attention UNet** architecture was chosen due to its enhanced ability to focus on salient spatial features, making it particularly suitable for multi-class segmentation in medical imaging.

#### Benefits:

- **Attention Gates**: Improve model focus on relevant regions  
- **Feature Refinement**: Suppresses irrelevant feature propagation  
- **Gradient Enhancement**: Encourages better learning of meaningful features  

This model strikes a balance between computational feasibility and segmentation accuracy.

---

## Implementation Strategy

- Resized input images to **256×256** for better efficiency without sacrificing spatial detail  
- Incorporated **attention mechanisms** in the UNet to dynamically prioritize relevant features  
- Designed to enhance **gradient flow** and reduce redundancy in deep layers  

---

## Summary

This dataset and modeling strategy support reproducible research in medical image segmentation. While providing a solid foundation for developing segmentation algorithms, this dataset should **not be used for real-world clinical applications** without further validation on up-to-date and diverse datasets.

---
