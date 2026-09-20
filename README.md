# Cross-City Building Age Classification from Satellite Imagery

## Overview

This project investigates whether satellite imagery can be used to classify building age in one city and transfer that information to another city.

The model is trained primarily on **Madrid** and evaluated on **Amsterdam**. The main challenge is not simply classifying pixels within one city, but handling **spatial dependence and domain shift** between cities.

Three issues are central:

- **Weak pixel-level signal:** 30 m pixels can contain buildings together with roads, vegetation, gardens and shadows.
- **Spatial dependence:** neighbouring pixels often contain related urban and building information.
- **Domain shift:** Madrid and Amsterdam have different urban environments and different age-class boundaries.

The project addresses these issues through improved temporal features, spatial neighbourhood features, spatially blocked validation and few-shot prototype transfer.

---

## Repository Structure

```text
.
├── baseline_problem/
│   └── Original problem notebooks and baseline implementation
│
├── 3NoteBook_Processing.ipynb
├── 4Notebook_Modelling.ipynb
└── README.md
```

### `baseline_problem/`

Contains the original notebooks and files **as they were provided for the problem**. This preserves the original baseline so the improvements can be compared against the starting point. It is worth reading these for a better baseline understand of how the problem was approached. This README is more of a shortened overview of the work done, the Notebooks in this folder are more verbose and informative.

### `3NoteBook_Processing.ipynb`

Improves the original preprocessing pipeline and produces the expanded **84-feature** representation.

### `4Notebook_Modelling.ipynb`

Contains the spatial feature engineering, modelling, validation, ablation experiments and Amsterdam transfer experiments.

---

# The Problem

The target is a four-class building-age classification problem derived from `weighted_mean_year`.

The class boundaries differ between the two cities:

| Class | Madrid | Amsterdam |
|---|---|---|
| 1 | Pre-1960 | Pre-1945 |
| 2 | 1960–1984 | 1945–1984 |
| 3 | 1984–2004 | 1984–2004 |
| 4 | 2004–2024 | 2004–2024 |

This creates a genuine transfer problem: a model trained on Madrid cannot simply assume that its learned class boundaries have the same meaning in Amsterdam.

The original representation contained **60 features** derived from Landsat time-series data.

---

# Notebook 3 — Processing & Feature Engineering

`3NoteBook_Processing.ipynb` extends the original 60-feature representation to **84 features**.

## 1. Temporal Contrast Features

For each relevant band:

```text
late_mean - early_mean
```

is calculated.

This gives the model an explicit measure of how the spectral signal changed over time rather than requiring it to infer the difference from separate early and late features.

**Added: 12 features**

## 2. Temporal Trend Features

An OLS slope is calculated against year for the six spectral bands and five spectral indices.

This captures the direction of change across the approximately 40-year time series.

**Added: 11 features**

## 3. Building Coverage

`coverage` is retained as an explicit model feature.

This gives the model information about how much of a pixel actually overlaps building footprints, helping distinguish strong building signals from pixels containing relatively little building area.

**Added: 1 feature**

Together:

```text
60 original features
+ 12 contrast features
+ 11 trend features
+ 1 coverage feature
= 84 features
```

The resulting representation is saved as `preprocessed_v2.pkl`.

Experiments with additional features such as SAVI, ENDISI and temporal breakpoint detection were tested but not retained in the final pipeline.

---

# Notebook 4 — Spatial Modelling & Transfer

`4Notebook_Modelling.ipynb` builds on the 84 processed features.

## Multi-Scale Spatial Context

Buildings are spatially structured, so individual pixels are supplemented with information from their surrounding pixels.

A `scipy.spatial.cKDTree` is used to efficiently calculate neighbourhood statistics at five scales:

```text
k = 16
k = 64
k = 256
k = 1024
k = 2048
```

Twelve informative source features are aggregated using:

- Neighbourhood mean
- Neighbourhood standard deviation

This produces:

```text
12 features × 5 scales × 2 statistics
= 600 spatial features
```

Combined with the 84 processed features:

```text
84 base features + 600 spatial features
= 684 model features
```

---

# Preventing Spatial Leakage

Randomly splitting pixels is inappropriate for this problem because nearby pixels are highly correlated.

The final pipeline therefore divides the city into contiguous **300 m spatial blocks** and uses `GroupKFold`.

Neighbourhood features for validation pixels are constructed from the training reference pixels rather than allowing validation pixels to become part of their own neighbourhood reference cloud.

This prevents the model from gaining an artificially optimistic score by seeing spatially adjacent validation information during training.

The final notebook uses **10-fold spatially blocked cross-validation** for its headline evaluation.

---

# Model

The Stage-2 classifier is **LightGBM** with balanced class weights.

However, the Amsterdam few-shot evaluation does not simply fine-tune this classifier.

The transfer pipeline instead uses the learned feature representation directly.

This distinction is important:

```text
Stage 1
84 temporal features
        +
600 spatial features
        ↓
684-dimensional representation

Stage 2
Madrid → LightGBM

Stage 3
Amsterdam labels → class prototypes
```

---

# Few-Shot Transfer

For Amsterdam, a small number of labelled pixels per class are selected.

A prototype is calculated for each class as the mean feature vector of its labelled examples:

```text
Prototype(class) = mean(feature vectors for that class)
```

Each unlabelled Amsterdam pixel is then assigned to the nearest class prototype using Euclidean distance.

The Stage-2 LightGBM classifier is **not used** for this prototype prediction step.

The method is evaluated at several target-label budgets:

```text
5, 25, 50, 100, 200 labels per class
```

Each scored budget is evaluated over multiple random support selections.

---

# Results

The development process produced the following headline results:

| Metric | Original Baseline | Intermediate | Final |
|---|---:|---:|---:|
| Madrid blocked CV Macro F1 | 0.6179 | 0.6849 | **0.7000** |
| Amsterdam zero-shot Macro F1 | 0.3427 | 0.4787 | **0.5265** |
| Amsterdam 25-shot Macro F1 | 0.5986 | 0.6340 | **0.6494** |

The final pipeline therefore improves both within-city performance and transfer performance.

The main lesson from the experiments is that **feature representation and spatial context were more important than simply changing the classifier**.

---

# Important Design Decisions

### Why stop at `k = 2048`?

A `k = 4096` neighbourhood was also tested. Although it produced a very small improvement on Madrid, transfer performance decreased.

At very large scales, neighbourhood statistics begin to resemble city-level averages. This can make the representation more city-specific and less transferable.

The final pipeline therefore uses:

```text
16 → 64 → 256 → 1024 → 2048
```

### Why use raw features for prototype transfer?

Both raw and re-standardised prototype spaces were tested.

For the final 84-feature representation, the raw augmented space performed better across the evaluated support budgets and was therefore retained.

---

# Results in Context

The final system combines:

```text
Landsat time-series information
            +
Temporal contrast and trend features
            +
Building coverage
            +
Multi-scale spatial context
            +
Spatially blocked validation
            +
Few-shot target-city adaptation
```

The resulting pipeline is designed not only to classify building age within the source city, but to explicitly test whether the learned representation remains useful when transferred to a different urban environment.

---

# Running the Project

Install the main dependencies:

```bash
pip install numpy pandas scipy scikit-learn lightgbm matplotlib jupyter
```

Then run the notebooks in order:

### 1. Processing

Open:

```text
3NoteBook_Processing.ipynb
```

This creates the processed 84-feature dataset.

### 2. Modelling

Open:

```text
4Notebook_Modelling.ipynb
```

This performs spatial feature construction, cross-validation, modelling, ablation experiments and Amsterdam transfer evaluation.

For understanding the development process, start with the original notebooks in:

```text
baseline_problem/
```

and then compare them with Notebooks 3 and 4.

---

## Technologies

- Python
- NumPy
- Pandas
- SciPy
- scikit-learn
- LightGBM
- Matplotlib
- Jupyter Notebook
- Landsat satellite imagery
