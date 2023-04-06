# COMP 6211H Deep Learning in Medical Image Analysis Final Project

## Background

- [x] Review dataset
- [ ] Decide what datatype to be used
- [ ] Paper review
- [ ] Run baseline model
- [ ] Develop based on existing model

## Timeline

| Item                            | Time     |
| :------------------------------ | :------- |
| Review dataset                  | 4-4-2023 |
| Decide what datatype to be used |          |
| Paper review                    |          |
| Run baseline model              |          |
| Develop based on existing model |          |

## Data

### Chest X-ray (Covid-19 & Pnumonia)

#### Basic Information of Chest X-ray

Data type : Chest X ray
| Set   | Label     | Number |
| ----- | --------- | ------ |
| Train | Normal    | 1266   |
|       | PNEUMONIA | 3418   |
|       | COVID-19  | 460    |
| Test  | Normal    | 317    |
|       | PNEUMONIA | 855    |
|       | COVID-19  | 116    |

Train-Test split = 8:2

Label Ratio

- COVID-19 : 576 (0.09)
- Normal : 1583 (0.24)
- PNEUMONIA : 4273 (0.66)

Image Size: ~2k*2k
Preprocess needed for grayscale as some of the image are in purple color.

### COVID-19_Radiography_Dataset

#### Basic Information of Radiography

Source: from different dataset, like Kaggle, Github, existing research paper and clinical support.
Data structure: With an image of 256*256, and a mask specify the lung region.

Data type : Chest X ray
| Label           | Number | Ratio |
| --------------- | ------ | ----- |
| Normal          | 10192  | 0.48  |
| Viral PNEUMONIA | 1345   | 0.06  |
| Lung Opacity    | 6012   | 0.28  |
| COVID-19        | 3616   | 0.17  |

### SARS-COV-2_Ct-Scan_Dataset

#### Basic Information of SARS

CT Scan, Baseline 97.31% F1 on eXplainable Deep Learning approach (xDNN)
