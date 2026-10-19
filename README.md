# CT Tumor Segmentation

End-to-end 3D CT tumor segmentation project developed in Python and PyTorch using medical imaging data from the Medical Segmentation Decathlon.

The project covers the complete workflow from NIfTI image loading and preprocessing to 3D deep learning, quantitative evaluation, lesion-wise analysis, visualization, and post-processing.

> This project is developed for educational and research purposes only.  
> It is not a medical device and must not be used for clinical diagnosis or treatment decisions.


## Overview

The goal of this project is to build and study a complete 3D medical image segmentation pipeline for liver tumor segmentation from CT scans.

The project currently includes:

- NIfTI image loading
- CT intensity inspection
- Hounsfield Unit windowing
- intensity normalization
- voxel-spacing resampling
- patient-level train/validation splitting
- 3D tumor-centered patch extraction
- data augmentation
- PyTorch Dataset and DataLoader
- custom 3D U-Net
- multiclass segmentation experiments
- binary tumor segmentation experiments
- combined segmentation losses
- model checkpointing
- voxel-wise evaluation
- lesion-wise evaluation
- connected-component post-processing
- visual error analysis
- CSV evaluation reporting


## Project Structure

```text
ct-tumor-segmentation/
│
├── configs/
│
├── data/
│   ├── sample/
│   │   ├── image/
│   │   └── label/
│   │
│   └── multicase/
│       ├── images/
│       └── labels/
│
├── docs/
│   └── images/
│
├── notebooks/
│
├── outputs/
│   ├── checkpoints/
│   └── evaluation/
│
├── src/
│   ├── analysis/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   └── visualization/
│
├── tests/
│
├── .gitignore
├── README.md
└── requirements.txt
```


## Dataset

The project uses the **Task03 Liver** dataset from the Medical Segmentation Decathlon.

The segmentation labels are:

```text
0 = Background
1 = Liver
2 = Tumor
```

The initial development was performed on `liver_0`, followed by a small multi-patient subset containing:

```text
liver_0
liver_1
liver_2
liver_3
liver_4
liver_5
liver_6
liver_7
liver_8
liver_9
```

The current experimental split uses:

```text
8 training patients
2 validation patients
```

The split is performed at patient level in order to avoid patient leakage between training and validation data.


## NIfTI Loading

Medical volumes are loaded using `nibabel`.

The loader returns:

```text
3D NumPy volume
+
NIfTI affine matrix
```

Example CT volume:

```text
Shape:
(512, 512, 75)

Intensity range:
-1024 HU to 1410 HU
```

Example segmentation mask:

```text
Shape:
(512, 512, 75)

Labels:
[0, 1, 2]
```


## Voxel Spacing

The original voxel spacing for the initial CT sample is approximately:

```text
0.703125 mm
0.703125 mm
5.000000 mm
```

This shows that the original volume is anisotropic, with much larger spacing along the slice direction.


## Tumor Slice Detection

The ground-truth segmentation mask is analyzed to identify the axial slice containing the largest number of tumor voxels.

For the initial sample:

```text
Maximum tumor slice:
58

Tumor voxels in slice:
538
```

This operation is based on the provided ground-truth annotation and is not an AI prediction.


## Tumor Visualization

The project includes visualization utilities for:

- CT slices
- tumor masks
- CT + tumor overlays
- 3D tumor-centered patches
- binary model predictions

Example visualization:

![Liver tumor visualization](docs/images/liver_0_tumor_overlay.png)


## Quantitative Tumor Analysis

Tumor volume can be estimated from:

```text
number of tumor voxels
×
physical voxel volume
```

For the initial sample:

```text
Voxel spacing:
[0.703125, 0.703125, 5.0] mm

Tumor voxels:
2591

Tumor volume:
6404.75 mm³

Tumor volume:
6.40 cm³
```

This volume is derived from the dataset ground-truth segmentation and image spacing.

It must not be interpreted as an independent clinical measurement.


## CT Intensity Windowing

CT images are represented in Hounsfield Units.

An example soft-tissue window is applied using:

```text
Window center:
50 HU

Window width:
150 HU
```

This corresponds to:

```text
Lower bound:
-25 HU

Upper bound:
125 HU
```

Values outside this range are clipped.

Example:

```text
Original range:
-1024 to 1410 HU

Windowed range:
-25 to 125 HU
```


## Intensity Normalization

After CT windowing, image intensities are normalized to approximately:

```text
0.0 to 1.0
```

This provides a more suitable input range for neural network training.


## Voxel-Spacing Resampling

CT images and segmentation masks are resampled to a common target spacing:

```text
1.0 × 1.0 × 2.5 mm
```

Example:

```text
Original shape:
(512, 512, 75)

Resampled shape:
(360, 360, 150)
```

Different interpolation strategies are used for images and labels to preserve segmentation classes.


## Patient-Level Train / Validation Split

Cases are automatically discovered from the dataset directory.

A reproducible patient-level split is generated using a fixed random seed.

Current experimental split:

```text
Training cases:
8

Validation cases:
2

Overlap:
0
```

This ensures that images from the same patient do not appear in both sets.


## 3D Tumor Patch Extraction

To reduce computational and memory requirements, smaller 3D patches are extracted from the resampled volumes.

Current patch size:

```text
96 × 96 × 64
```

The tumor center is calculated from the ground-truth mask using:

```text
label = 2
```

A corresponding CT patch and segmentation patch are extracted around the same center.

Example:

```text
Tumor center:
(222, 168, 123)

Image patch:
(96, 96, 64)

Label patch:
(96, 96, 64)
```

The extracted patch preserves the original segmentation classes:

```text
[0, 1, 2]
```


## Multi-Patch Tumor-Aware Sampling

A single tumor-centered patch per patient was initially used.

To increase spatial variability, multiple shifted centers are now generated around the tumor center.

Current configuration:

```text
5 patches per training patient
8 training patients
40 training patches
```

The spatial shifts remain close to the tumor center so that each extracted training patch still contains tumor voxels.

Example for `liver_0`:

```text
Patch 1: 1334 tumor voxels
Patch 2: 1154 tumor voxels
Patch 3: 1334 tumor voxels
Patch 4: 1228 tumor voxels
Patch 5: 1246 tumor voxels
```


## Data Augmentation

To increase variability in the training data and reduce overfitting, data augmentation is applied to the extracted 3D patches.

The current implementation includes spatial 3D flipping.

The same transformation is applied to both:

```text
CT image patch
+
segmentation mask
```

in order to preserve image-label alignment.

The augmentation preserves:

- patch dimensions
- segmentation class values
- tumor voxel count

More advanced augmentation strategies may be introduced in later experiments.


## Current Preprocessing Pipeline

The current preprocessing workflow is:

```text
NIfTI loading
        ↓
CT windowing
        ↓
intensity normalization
        ↓
voxel-spacing resampling
        ↓
patient-level split
        ↓
tumor center detection
        ↓
multi-patch extraction
        ↓
data augmentation
        ↓
PyTorch Dataset
        ↓
DataLoader
```


## PyTorch Dataset and DataLoader

The preprocessed NumPy patches are converted into PyTorch tensors.

Image tensor shape:

```text
[1, 96, 96, 64]
```

Label tensor shape:

```text
[96, 96, 64]
```

Using a DataLoader with batch size 2 produces:

```text
Image batch:
[2, 1, 96, 96, 64]

Label batch:
[2, 96, 96, 64]
```


## 3D U-Net

A custom 3D U-Net architecture is implemented in PyTorch.

Example input:

```text
[2, 1, 96, 96, 64]
```

For the original three-class segmentation experiment:

```text
Output:
[2, 3, 96, 96, 64]
```

where the three output channels represent:

```text
0 = Background
1 = Liver
2 = Tumor
```


## Segmentation Loss

The multiclass training experiments use a combination of:

```text
Cross Entropy Loss
+
Dice Loss
+
Tumor-specific Dice Loss
```

Class weighting was explored because the tumor represents only a small fraction of the training voxels.

For the multi-patch training set, the approximate class distribution is:

```text
Background:
48.19%

Liver:
46.71%

Tumor:
5.10%
```


## Multiclass Training Experiments

The multiclass 3D U-Net successfully reduced training and validation loss.

However, tumor Dice remained approximately zero even after:

- class weighting
- tumor-specific Dice loss
- multiple tumor-aware patches
- longer training

Additional probability-based analysis showed that the model assigned higher tumor probability to real tumor voxels than to non-tumor voxels, but the tumor class rarely won the final multiclass `argmax`.

This motivated a controlled binary segmentation experiment.


## Binary Tumor Segmentation

The segmentation problem was reformulated as:

```text
0 = Non-tumor
1 = Tumor
```

Original labels were converted as:

```text
0 → 0
1 → 0
2 → 1
```

The number of tumor voxels is preserved during conversion.

The same 3D U-Net architecture is reused with:

```python
num_classes=2
```

The binary training loss combines:

```text
Weighted Cross Entropy
+
Tumor Dice Loss
```


## Binary Multi-Patient Training

The binary model was trained using:

```text
8 training patients
40 tumor-aware training patches
2 validation patients
10 epochs
```

At epoch 10:

```text
Train Tumor Dice:
0.1322

Validation Tumor Dice:
0.2148
```

The binary formulation therefore produced the first clearly non-zero hard tumor Dice on held-out validation data.


## Model Checkpointing

Two checkpoint strategies are supported:

```text
Best validation loss
```

and:

```text
Best validation Tumor Dice
```

This distinction is important because the epoch with the lowest validation loss is not necessarily the epoch with the best tumor segmentation performance.


## Phase 4 — Evaluation

The binary tumor segmentation model was evaluated on a patient-level validation split using both voxel-wise and lesion-wise metrics.

The current validation subset contains only two patients and is therefore intended as an exploratory technical evaluation rather than a reliable estimate of clinical performance.


### Voxel-Wise Metrics

The following metrics are currently implemented:

- Dice score
- Intersection over Union (IoU / Jaccard)
- Precision
- Recall / Sensitivity

Using the checkpoint selected by validation Tumor Dice, the validation cases showed markedly different behavior.

| Case | Dice | IoU | Precision | Recall | GT Tumor Voxels | Predicted Tumor Voxels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| liver_3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 529 | 0 |
| liver_7 | 0.4296 | 0.2735 | 0.9875 | 0.2745 | 6350 | 1765 |

The results highlight substantial case-to-case variability.

For `liver_3`, the lesion is completely missed.

For `liver_7`, the model localizes part of the tumor with very high precision but relatively low recall, indicating a conservative segmentation that captures only part of the lesion volume.


### Lesion-Wise Evaluation

Connected-component analysis with 3D 26-connectivity was introduced to evaluate tumor detection at lesion level.

| Case | GT Lesions | Predicted Lesions | Detected | Missed | False Positive Lesions | Lesion Sensitivity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| liver_3 | 1 | 0 | 0 | 1 | 0 | 0.0000 |
| liver_7 | 3 | 6 | 2 | 1 | 3 | 0.6667 |

This analysis provides information that complements voxel-wise metrics.

A model may only partially segment a lesion while still correctly detecting it as an object.


### Lesion Size Analysis

Connected-component sizes were analyzed to distinguish substantial predictions from very small isolated fragments.

For `liver_7`, the ground-truth lesions contained:

```text
GT lesion 1: 4796 voxels
GT lesion 2: 1537 voxels
GT lesion 3:   17 voxels
```

The raw prediction contained:

```text
Pred lesion 1:   19 voxels
Pred lesion 2:   40 voxels
Pred lesion 3: 1381 voxels
Pred lesion 4:  323 voxels
Pred lesion 5:    1 voxel
Pred lesion 6:    1 voxel
```

This analysis showed that several predicted components were extremely small and likely represented isolated segmentation fragments.


### Connected-Component Post-Processing

Small predicted connected components were analyzed as possible false-positive fragments.

An exploratory minimum connected-component size of:

```text
20 voxels
```

was used as a post-processing filter.

For `liver_7`:

| Metric | Raw | Filtered |
| --- | ---: | ---: |
| Dice | 0.4296 | 0.4307 |
| IoU | 0.2735 | 0.2744 |
| Precision | 0.9875 | 0.9994 |
| Recall | 0.2745 | 0.2745 |
| Predicted lesions | 6 | 3 |
| Detected lesions | 2 | 2 |
| Missed lesions | 1 | 1 |
| False-positive lesions | 3 | 0 |
| Lesion sensitivity | 0.6667 | 0.6667 |

The post-processing step removed small false-positive components without reducing recall or lesion sensitivity.

The minimum lesion-size threshold was selected during exploratory analysis and must not be interpreted as a clinically validated parameter.


### Error Analysis

Visual inspection confirmed two different failure modes.

#### Validation Case — liver_3

The binary model completely misses the tumor in this validation case.

![Binary tumor segmentation - liver_3](docs/images/liver_3_binary_prediction.png)

#### Validation Case — liver_7

The model correctly localizes part of the tumor but substantially under-segments the lesion.

![Binary tumor segmentation - liver_7](docs/images/liver_7_binary_prediction.png)

These experiments demonstrate why decreasing validation loss alone is not sufficient to assess medical image segmentation quality.

Voxel-wise metrics, lesion-wise metrics, connected-component analysis, visual inspection, and per-case analysis provide complementary information.

All evaluation results are exploratory and are based on a very small validation subset.


## Evaluation Outputs

Structured evaluation results are saved as CSV files.

Current outputs include:

```text
outputs/evaluation/binary_validation_metrics.csv
outputs/evaluation/binary_validation_raw_vs_filtered.csv
```

These files contain patient-level metrics and allow reproducible comparison of raw and post-processed predictions.


## Current Python Stack

The project currently uses:

```text
Python
NumPy
NiBabel
Matplotlib
SciPy
PyTorch
```

Additional libraries may be introduced in later phases for medical imaging workflows and advanced deep learning pipelines.


## Current Features

- [x] NIfTI loading
- [x] CT volume inspection
- [x] segmentation mask loading
- [x] voxel-spacing analysis
- [x] CT slice visualization
- [x] tumor mask overlay
- [x] tumor volume calculation
- [x] CT intensity windowing
- [x] intensity normalization
- [x] voxel-spacing resampling
- [x] patient-level train/validation split
- [x] 3D tumor-centered patch extraction
- [x] multi-patch tumor-aware sampling
- [x] 3D data augmentation
- [x] PyTorch Dataset
- [x] PyTorch DataLoader
- [x] custom 3D U-Net
- [x] multiclass loss
- [x] tumor-aware loss
- [x] optimizer and backpropagation
- [x] training loop
- [x] validation loop
- [x] checkpoint saving/loading
- [x] multiclass segmentation experiments
- [x] binary tumor segmentation
- [x] Dice evaluation
- [x] IoU/Jaccard evaluation
- [x] Precision evaluation
- [x] Recall/Sensitivity evaluation
- [x] per-case evaluation
- [x] visual error analysis
- [x] lesion-wise evaluation
- [x] lesion-size analysis
- [x] connected-component post-processing
- [x] CSV evaluation reporting


## Roadmap

### Phase 1 — Medical Image Exploration

- [x] Load NIfTI CT volumes
- [x] Load segmentation masks
- [x] Inspect voxel spacing
- [x] Visualize CT slices
- [x] Overlay tumor annotations
- [x] Calculate tumor volume


### Phase 2 — Preprocessing

- [x] CT intensity windowing
- [x] intensity normalization
- [x] voxel-spacing resampling
- [x] train/validation split
- [x] 3D patch extraction
- [x] tumor-aware multi-patch sampling
- [x] data augmentation


### Phase 3 — Deep Learning

- [x] PyTorch Dataset
- [x] DataLoader
- [x] custom 3D U-Net
- [x] Dice-based losses
- [x] weighted Cross Entropy
- [x] training pipeline
- [x] validation pipeline
- [x] checkpoint saving/loading
- [x] multiclass segmentation experiments
- [x] binary tumor segmentation experiments


### Phase 4 — Evaluation

- [x] Dice score
- [x] IoU / Jaccard
- [x] Precision
- [x] Recall / Sensitivity
- [x] per-case evaluation
- [x] probability and threshold analysis
- [x] visual error analysis
- [x] lesion-wise evaluation
- [x] connected-component analysis
- [x] lesion-size analysis
- [x] post-processing evaluation
- [x] CSV reporting


### Phase 5 — Clinical Imaging Workflow

- [ ] DICOM loading
- [ ] DICOM series reconstruction
- [ ] Hounsfield Unit handling from DICOM metadata
- [ ] clinical-style preprocessing
- [ ] inference on DICOM volumes
- [ ] DICOM SEG / RTSTRUCT exploration


## Limitations

The current experimental results are based on a very small subset of the available dataset.

In particular:

```text
10 patients total
8 training patients
2 validation patients
```

The reported metrics therefore have high statistical uncertainty and must not be interpreted as evidence of clinical model performance.

The current patch-based workflow is also tumor-centered using ground-truth information during training and experimental validation preprocessing.

Future work should evaluate more realistic inference strategies on complete volumes and substantially larger patient cohorts.


## Disclaimer

This project is developed for educational and research purposes only.

It is not a medical device and must not be used for clinical diagnosis, treatment planning, or medical decision-making.