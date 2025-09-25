# Exploring Multimodal Approaches in Computational Pathology

## Project Background
This research project was conducted at the **TIA Centre (Tissue Image Analytics)**, University of Warwick, under the supervision of Dr. Adam Shephard and Dr. Nasir Rajpoot, with funding provided by the **Undergraduate Research Support Scheme (URSS)**. 

## Overview
This project investigates whether **combining radiology and histology data** through **multimodal deep learning** approaches can improve predictions of time to biochemical recurrence (BCR) following prostatectomy compared to **unimodal approaches**. Unimodal architectures using histology and radiology images alone were benchmarked against multimodal architectures using both modalities together, using time-to-event analysis with **C-Index** as the primary metric of comparison. <br />

To evaluate whether histology-radiology multimodal models can outperform unimodal models in predicting time to BCR, we first developed a unimodal architecture that uses only histology or radiology images, followed by multimodal architectures that uses both modalities simultaneously. We then benchmarked the best performing unimodal architecture against the best performing multimodal architecture for this task.

## Dataset
We used the CHIMERA dataset ([CHIMERA Challenge](https://chimera.grand-challenge.org/task-1-prostate-cancer-biochemical-recurrent-prediction/)) containing:
- 95 cases with radiology, histology, and clinical data
- 27 positive and 68 negative BCR cases
- Matched whole slide images (WSI) and MRI scans

Licensed under CC-BY-NC-SA. See Citation section for full details.
 
## Unimodal Architectures
<img width="1145" height="345" alt="Screenshot 2025-09-23 at 21 33 26" src="https://github.com/user-attachments/assets/ca5687de-5ead-4c81-a096-23c3daaacf1c" />

Histology and radiology images were passed through foundational models to get vector representations of the images. These vector representations were then passed into a predictive MLP architecture for time to BCR predictions. Extensive hyperparameter tuning was carried out using cross-validation to ensure best time to BCR prediction results. <br />

To get the best possible unimodal results for benchmarking against multimodal approaches, we tested different histology and radiology foundation models to find the best performing encoder for each modality. We then used the best performing encoder for each modality in our multimodal pipeline, allowing us to benchmark multimodal models against the best performing unimodal models. <br />

#### Radiology Encoders (3D):
- MedicalNet (Best performer: 0.5584 ± 0.1250 C-Index)
- nnU-net
- Radiomics

We developed a comprehensive testing framework for histology encoders, evaluating all combinations of magnification levels (10x, 20x) and patch sizes (256, 512 pixels). The following histology foundation models were tested.

#### Histology Encoders (Slide-level): 
- PRISM (Best performer: 0.8181 ± 0.1018 C-Index)
- CHIEF
- TITAN
- MADELEINE <br />

_Histology foundation models obtained from the [Mahmood Lab TRIDENT repository](https://github.com/mahmoodlab/TRIDENT)_



## Multimodal Architectures
Contemporary fusion methodologies in medical imaging include early fusion, intermediate fusion, and late fusion methodologies. This project implements **intermediate fusion methodologies** to combine histology and radiology images.

## Marginal Intermediate Fusion
Marginal intermediate fusion refers to combining vector representations of each modality **with no additional learning**​.

<img width="1197" height="448" alt="Screenshot 2025-09-23 at 21 34 55" src="https://github.com/user-attachments/assets/c4cacc08-8870-4206-9f93-c39aa26fa11a" />

After getting vector representations of each modality, the vectors were concatenated and passed through a predictive MLP architecture for time to BCR predictions.

## Joint Intermediate Fusion
Joint intermediate fusion refers to combining vector representations of each modality **with additional learning**​.

<img width="1060" height="436" alt="Screenshot 2025-09-23 at 21 47 57" src="https://github.com/user-attachments/assets/c161c1da-7b41-458d-b00b-ebe16454d4ba" />

After getting vector representations of each modality, **cross-attention** is run on both histology and radiology vectors to give both vectors context based on the vector representation of the other modality. These vectors are passed through a cross-attention attention layer and updated accordingly. <br /> 

After components of radiology and histology vectors are updated, **self-attention** is run on both histology and radiology vectors separately, allowing each modality to learn which of its components are most important for predicting time to BCR. These vectors are passed through a self-attention layer and updated accordingly. <br />  

After both histology and radiology vectors have been updated, they are combined via simple concatenation (similar to marginal intermediate fusion) or by calculating a bilinear product of both vectors. This combined vector is then passed through a predictive MLP architecture for time to BCR predictions.


## Overall Results & Findings
### Unimodal Results (using best performing histology and radiology foundation models) 
| Unimodal Model | Average C-Index (10 repeats) |
|:----------------:|:------------------------------:|
| PRISM 20x 256 (Histology) | 0.8181 ± 0.1018 |
| MedicalNet (Radiology) | 0.5584 ± 0.1250 |

### Multimodal Results
| Fusion methodology | Average C-Index (10 repeats) |
|:----------------:|:------------------------------:|
| Marginal Intermediate Fusion | 0.7520 ± 0.1069​ |
| Joint Intermediate Fusion | 0.7420 ± 0.1114​ |

- **Radiology features alone** were **not a good predictor** for time to BCR, with an average C-Index score of 0.55 across 2 encoders
- Both Marginal and Joint Intermediate Fusion **did not yield better results** compared to unimodal approaches, with histology alone being a better predictor​
- Despite radiology's clinical relevance, its weak predictive power (C-index = 0.55)  **degraded the performance** of histology-based models rather than providing complementary information
- Incorporating modalities with poor individual predictive performance in multimodal approaches **can be counterproductive**

## Citations
We used the **CHIMERA dataset** from the CHIMERA Challenge: Combining histology, radiology and molecular data for medical prognosis and diagnosis.
```bibtex
@misc{chimera_challenge_2025,
  title={CHIMERA: Combining Histology, Medical imaging (radiology) and molecular data for medical prognosis and diagnosis challenge},
  year={2025},
  howpublished={Grand Challenge},
  url={https://chimera.grand-challenge.org/},
  note={Task 1: Prostate Cancer Biochemical Recurrence Prediction}
}
```

This work utilises pretrained histopathology foundation models from the Mahmood Lab TRIDENT repository:
```bibtex
@software{mahmood_lab_trident,
  title={TRIDENT: Towards Reliable multimodal Integrative DiagnosticE Network Technologies},
  author={Mahmood Lab},
  url={https://github.com/mahmoodlab/TRIDENT},
  year={2024},
  note={Foundation models: PRISM, CHIEF, TITAN, MADELEINE}
}

@article{prism2024,
  title={A General-Purpose Self-Supervised Model for Computational Pathology},
  author={Vorontsov, Eugene and Bozkurt, Alican and Casson, Adam and Shaikovski, George and Zelechowski, Michal and Severson, Kristen and Shaham, Eric and Gao, Sheng and Guo, Hao and Bai, Siqi and others},
  journal={arXiv preprint arXiv:2308.15474},
  year={2024}
}

@article{chen2024chief,
  title={A General-Purpose Multimodal Foundation Model for Dermatology},
  author={Chen, Richard J and Ding, Tong and Lu, Ming Y and Williamson, Drew FK and Jaume, Guillaume and Chen, Bowen and Zhang, Andrew and Shao, Daniel and Song, Andrew H and Shaban, Muhammad and others},
  journal={arXiv preprint arXiv:2407.03496},
  year={2024}
}

@article{titan2023,
  title={A Multimodal Foundation Model for Cancer Imaging Biomarker Discovery},
  author={Lu, Ming Y and Chen, Bowen and Williamson, Drew FK and Chen, Richard J and Zhao, Ivy and others},
  journal={bioRxiv},
  year={2023}
}
```

## Acknowledgments
- TIA Centre (Tissue Image Analytics) for institutional support
- CHIMERA Challenge organisers and data contributors
- Mahmood Lab for providing pretrained foundation models via the TRIDENT repository
- Pre-trained radiology model authors (MedicalNet, nnU-net)

## Contact
**Seth Alain Chang** - [https://www.linkedin.com/in/seth-alain-chang/]  <br />  
**Muhammad Amjad** - [https://www.linkedin.com/in/m4mjad/]




