# Clustering-Based Analysis of Colorectal Tissue Patches

A comparative study of multiple feature representations and clustering algorithms applied to colorectal cancer histopathology image patches.

---

## Overview

Digital pathology converts traditional microscope slides into whole slide images (WSIs) that can exceed 20 GB and reach resolutions of ~100,000 × 100,000 pixels. To make automated analysis tractable, WSIs are divided into smaller image patches, which can then be clustered to reveal underlying tissue structure.

This project evaluates unsupervised clustering pipelines on **5,000 colorectal cancer tissue patches** across **9 tissue types**, comparing how different combinations of feature representation, dimensionality reduction, and clustering algorithm affect the quality of the resulting clusters.

---

## Repository Structure

```
├── case_study_2_final.ipynb     # Main combined pipeline (PathologyGAN + ResNet50)
├── pathalogyGAN.ipynb           # PathologyGAN-specific exploration and clustering
├── ResNet50.ipynb               # ResNet50-specific exploration and clustering
├── Report_case2_GroupZ.pdf      # Full written report
├── Delta_Score_case2_GroupZ.pdf # Group contribution scores
└── colon_nct_feature/
    ├── pge_dim_reduced_feature.h5        # PathologyGAN PCA + UMAP features
    └── resnet50_dim_reduced_feature.h5   # ResNet50 PCA + UMAP features
```

> **Note:** The `colon_nct_feature/` folder containing the HDF5 feature files is not included in this repository due to file size. See the [Dataset](#dataset) section below for details.

---

## Dataset

**Source:** 5,000 non-overlapping image patches extracted from hematoxylin & eosin (H&E) stained histological images of human colorectal cancer and normal tissue.

Each patch is 224 × 224 × 3 pixels. Ground-truth tissue type labels are provided for all 5,000 patches across 9 categories:

| Abbreviation | Tissue Type |
|---|---|
| ADI | Adipose |
| BACK | Background |
| DEB | Debris |
| LYM | Lymphocytes |
| MUC | Mucus |
| MUS | Smooth muscle |
| NORM | Normal colon mucosa |
| STR | Cancer-associated stroma |
| TUM | Colorectal adenocarcinoma epithelium |

The original images are not used directly. Pre-extracted deep learning features are stored in HDF5 format, with both PCA-reduced and UMAP-reduced versions available for each feature extractor.

---

## Feature Representations

Two pretrained deep learning models were used for feature extraction:

### PathologyGAN (PGE)
A domain-specific GAN-based model trained on unlabelled histopathology images. It produces a **200-dimensional embedding** per patch that captures texture and structural patterns specific to pathology, which general-purpose CNNs are not trained to encode. Features are stored under the `pca_feature` and `umap_feature` keys in `pge_dim_reduced_feature.h5`.

### ResNet50
A 50-layer residual convolutional neural network pretrained on ImageNet. Its skip connections prevent gradient loss and allow the network to learn hierarchical visual features from natural images. ResNet50 produces a **2,048-dimensional feature vector** per patch. Features are stored in `resnet50_dim_reduced_feature.h5`.

---

## Dimensionality Reduction

Both feature sets are reduced to **100 dimensions** using two complementary methods before clustering:

### PCA — Principal Component Analysis
A linear technique that reorients the data along axes of maximum variance. The first principal component captures the most variance, with each subsequent component capturing the next highest. PCA is fast and interpretable, making it a reliable baseline, though it cannot capture non-linear structure in the data.

### UMAP — Uniform Manifold Approximation and Projection
A non-linear technique that preserves local neighbourhood relationships between data points. UMAP is more computationally intensive than PCA but produces more compact, visually separated cluster structures by capturing the manifold geometry of the data. It is particularly effective for visualising high-dimensional embeddings.

This gives **four feature representations** in total:

| ID | Feature Extractor | Dimensionality Reduction |
|----|-------------------|--------------------------|
| PGE-PCA | PathologyGAN | PCA → 100D |
| PGE-UMAP | PathologyGAN | UMAP → 100D |
| RN-PCA | ResNet50 | PCA → 100D |
| RN-UMAP | ResNet50 | UMAP → 100D |

---

## Clustering Algorithms

### Gaussian Mixture Model (GMM)
A probabilistic algorithm that models data as a mixture of Gaussian distributions. Each component has its own mean and covariance, allowing GMM to fit elliptical, overlapping, and concentric cluster shapes — more flexible than hard-assignment methods like K-Means. Parameters are estimated via the Expectation-Maximization (EM) algorithm.

```python
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=3, random_state=0)
labels = gm.fit_predict(features)
```

### Agglomerative (Hierarchical) Clustering
A bottom-up hierarchical approach. Each data point starts as its own cluster; the algorithm iteratively merges the two most similar clusters until the target number is reached. Unlike GMM or K-Means, it does not require any assumptions about cluster shape, and its dendrogram output allows exploration at multiple levels of granularity.

```python
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3)
labels = ac.fit_predict(features)
```

Both algorithms are run with `n_clusters = 3` / `n_components = 3`.

> The `pathalogyGAN.ipynb` and `ResNet50.ipynb` notebooks also include **K-Means** and **Louvain Clustering** as additional comparisons during exploratory work, though the main study focuses on GMM and Agglomerative Clustering.

---

## Evaluation Metrics

### Silhouette Score (internal)
Measures how similar each sample is to its own cluster compared to other clusters, without using ground-truth labels. Ranges from **−1 to +1**, where values close to +1 indicate well-separated, cohesive clusters. Values near 0 suggest overlapping boundaries; negative values indicate misassignment.

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(features, predicted_labels)
```

### V-measure (external)
An external metric that evaluates clustering quality against ground-truth tissue labels. It is the harmonic mean of two components:

- **Homogeneity** — each cluster contains only samples from a single tissue class
- **Completeness** — all samples of a given tissue class are assigned to the same cluster

Ranges from **0 to 1**, where 1 is a perfect match with the ground-truth labels.

```python
from sklearn.metrics import v_measure_score
score = v_measure_score(true_labels, predicted_labels)
```

---

## Results

All experiments sample 200 patches (`random_state=0`) from the full 5,000-patch dataset for clustering and evaluation.

### PathologyGAN Performance

| Metric | GaussianMixture | AgglomerativeClustering |
|--------|----------------|------------------------|
| Silhouette | 0.1508 | 0.1636 |
| V-measure | 0.3027 | 0.3232 |

### ResNet50 Performance

| Metric | GaussianMixture | AgglomerativeClustering |
|--------|----------------|------------------------|
| Silhouette | 0.1510 | 0.1713 |
| V-measure | 0.4797 | 0.3700 |

**Key observations:**

- ResNet50 + GMM achieved the highest V-measure (0.480), suggesting stronger alignment between cluster assignments and true tissue labels when using CNN features with probabilistic clustering.
- Agglomerative Clustering produced consistently higher Silhouette scores, indicating slightly better geometric separation of clusters.
- PathologyGAN features yielded lower V-measure scores overall, which may reflect that the domain-specific embeddings capture textural similarity rather than the class-discriminative features needed to separate all 9 tissue types.
- UMAP generally produces more compact, visually interpretable cluster patterns than PCA, though both are reduced to 100D before clustering.

---

## Visualisations

Each notebook generates:

**3D scatter plots** (via Plotly) of the first 3 PCA or UMAP components, coloured by ground-truth tissue type — used to visually assess how well tissue types naturally separate in each feature space before clustering.

**Stacked bar charts** (via Matplotlib) showing the percentage of each tissue type within each cluster, for each algorithm × feature combination — used to interpret what biological content each cluster represents.

---

## How to Run

### 1. Install dependencies

```bash
pip install h5py numpy pandas scikit-learn scikit-network matplotlib plotly
```

### 2. Set data paths

Update the HDF5 file paths in each notebook to point to your local copy of `colon_nct_feature/`:

```python
# In case_study_2_final.ipynb and pathalogyGAN.ipynb
pge_path = 'colon_nct_feature/pge_dim_reduced_feature.h5'

# In ResNet50.ipynb
resnet50_path = 'colon_nct_feature/resnet50_dim_reduced_feature.h5'
```

### 3. Run notebooks

Open and run the notebooks in order:

| Notebook | Purpose |
|---|---|
| `pathalogyGAN.ipynb` | PathologyGAN feature loading, 3D visualisation, K-Means / GMM / Agglomerative / Louvain clustering |
| `ResNet50.ipynb` | ResNet50 feature loading, 3D visualisation, K-Means / GMM / Agglomerative clustering |
| `case_study_2_final.ipynb` | Combined pipeline comparing both feature sets with GMM and Agglomerative Clustering, final evaluation tables and cluster composition charts |

---

## Dependencies

| Package | Purpose |
|---|---|
| `h5py` | Reading HDF5 feature files |
| `numpy` | Array operations |
| `pandas` | Tabular data and result formatting |
| `scikit-learn` | GMM, Agglomerative Clustering, K-Means, Silhouette, V-measure |
| `scikit-network` | Louvain Clustering (exploratory notebooks) |
| `matplotlib` | Cluster composition bar charts |
| `plotly` | Interactive 3D scatter plots |

---
