# Thermoelectric Figure of Merit (ZT) Prediction using XGBoost and Magpie Features

Author :- Subhajit Khan

## Overview

This project aims to predict the thermoelectric figure of merit (ZT) of materials using machine learning. The workflow follows a data-driven materials informatics approach inspired by recent research on thermoelectric property prediction.

The project uses:

* Material compositions (chemical formulas)
* Temperature
* Magpie descriptors generated using Matminer
* XGBoost regression model
* SHAP explainability analysis
* Learning curves 

The objective is to predict ZT values and identify the most influential material properties affecting thermoelectric performance.

---

## Dataset

### Original Dataset

Intially the data has taken from different sources for different parmeters like electrical resistivity, thermal conductivity, seeback coefficient and ZT for different tempretures.

Then The cleaned dataset contains:

* Chemical Formula
* Temperature (K)
* ZT (Target Variable)

Example:

| Formula             | Temperature | ZT    |
| ------------------- | ----------- | ----- |
| Pb0.7Mn0.3Te1Na0.04 | 350         | 0.276 |
| Pb0.7Mn0.3Te1Na0.04 | 400         | 0.420 |
| Pb0.7Mn0.3Te1Na0.04 | 450         | 0.564 |

---

## Project Workflow

### Step 1: Data Preparation

Performed data cleaning and preprocessing:

* Removed invalid entries
* Removed duplicates
* Standardized chemical formulas
* Retained:

  * Formula
  * Temperature
  * ZT

Output:

```text
Cleaned_Data.csv
```

---


### step 2: Distribution of Target Variable

![violin_plot](Figures/violin_plot_ZT_values.png)

**Insights:**
- Shows the distribution of zT values.
- Highlights skewness and outliers.
- Reveals density of observations across performance ranges.

--- 

### Step 3: Feature Engineering using Matminer

Chemical formulas were converted into pymatgen Composition objects.

```python
df["composition"] = df["Formula"].apply(Composition)
```

Magpie descriptors were generated using:

```python
ElementProperty.from_preset("magpie")
```

Output:

```text
Magpie_Features.csv

```

Generated features include:

* Atomic Weight
* Electronegativity
* Covalent Radius
* Melting Temperature
* Valence Electron Information
* Space Group Statistics

Result:

133 composition-based features

Temperature was added as an additional feature.

---

### Step 4: Missing Value Handling

Missing values were handled using median imputation.

```python
X = X.fillna(X.median())
```

This preserved all available samples.

---

### Step 5: Compound-wise Train/Test Split

To avoid data leakage, splitting was performed at the compound level.

Instead of splitting rows randomly:

```text
PbTe @ 300K
PbTe @ 500K
```

the entire compound was assigned to either train or test.

This ensures evaluation on unseen compounds.

---

### Step 6: Baseline XGBoost Model

A baseline XGBoost regressor was trained.

Initial Results:

```text
Train R² = 0.9949943423271179
Test R² = 0.6360927820205688

```

Strong overfitting was observed.

---

### Step 7: Hyperparameter Optimization

RandomizedSearchCV was used for tuning.

Parameters searched:

* max_depth
* learning_rate
* n_estimators
* reg_alpha
* reg_lambda
* min_child_weight
* subsample
* colsample_bytree

Best model improved performance but overfitting remained.

Results:

```text
Best Parameters 
 max_depth = 5
 learning_rate = 0.1
 n_estimators = 500
 reg_alpha = 3
 reg_lambda = 10
 min_child_weight = 5
 subsample = 0.6
 colsample_bytree = 0.6

Best score = 0.7571158766746521

Train R² =  0.9436091780662537
Test R² = 0.6923245191574097
```

---

### Step 8: Feature Selection

Feature importance scores were extracted from XGBoost.

Different subsets were tested:

* Top 25 Features
* Top 50 Features
* Top 75 Features
* All Features

Best performance was achieved using:

```text
Top 50 Features : top50_features.pkl
```

Results:

```text
Train R² = 0.9416772723197937
Test R² = 0.7147465348243713

```

---

### Step 9: Cross Validation

5-Fold Cross Validation was performed.

Results:

```text
Fold Scores:
0.632
0.812
0.792
0.794
0.731
```
![5_Fold_cross_validation_plot](Figures/cross_validation_scores.png)

Performance:

```text
Mean R² = 0.752
Std = 0.066
```

The model demonstrated good stability and generalization.

---

### Step 10: SHAP Analysis

SHAP analysis was performed to understand feature importance.

Outputs:

* ![SHAP Summary Plot](Figures/SHAP_Summary.png)

* Feature Importance Analysise Ranking

The figure below highlights the 20 most influential features identified by the XGBoost model for predicting thermoelectric performance.

| Rank | Feature                              | Importance |
| ---- | ------------------------------------ | ---------: |
| 1    | Temperature                          |   0.210463 |
| 2    | MagpieData mean GSvolume_pa          |   0.033905 |
| 3    | MagpieData avg_dev CovalentRadius    |   0.033621 |
| 4    | MagpieData avg_dev MendeleevNumber   |   0.026665 |
| 5    | MagpieData avg_dev GSvolume_pa       |   0.026160 |
| 6    | MagpieData avg_dev Number            |   0.015770 |
| 7    | MagpieData mean MeltingT             |   0.015300 |
| 8    | MagpieData mean MendeleevNumber      |   0.014717 |
| 9    | MagpieData mean SpaceGroupNumber     |   0.014174 |
| 10   | MagpieData avg_dev SpaceGroupNumber  |   0.013987 |
| 11   | MagpieData avg_dev Row               |   0.013021 |
| 12   | MagpieData mean NpValence            |   0.011975 |
| 13   | MagpieData range MeltingT            |   0.011771 |
| 14   | MagpieData range MendeleevNumber     |   0.011302 |
| 15   | MagpieData mean NValence             |   0.010924 |
| 16   | MagpieData avg_dev NUnfilled         |   0.009955 |
| 17   | MagpieData range GSvolume_pa         |   0.009570 |
| 18   | MagpieData maximum NpUnfilled        |   0.008763 |
| 19   | MagpieData minimum NUnfilled         |   0.008511 |
| 20   | MagpieData avg_dev Electronegativity |   0.008296 |

#### Key Insights

* **Temperature** is the dominant predictor, contributing over **21%** of the model's total importance.
* Features related to **atomic volume (GSvolume_pa)** and **covalent radius variation** are among the most influential compositional descriptors.
* **Mendeleev Number**-based features appear multiple times, indicating the importance of elemental chemical trends in thermoelectric performance.
* Electronic structure descriptors such as **NpValence**, **NValence**, and **NUnfilled** also contribute significantly to prediction accuracy.
* Structural descriptors including **SpaceGroupNumber** provide additional information about material behavior.

These results suggest that thermoelectric performance is governed by a combination of temperature, atomic-scale structural properties, and electronic characteristics of constituent elements.


---

### Step 11: Learning Curve Analysis

Learning curves were generated:

* ![R² vs Training Dataset Size](Figures/LearningCurve_R2.png)
* ![MAE vs Training Dataset Size](Figures/LearningCurve_MAE.png)

These curves help determine whether additional training data could improve model performance.

---

## Final Model Performance

### ### Actual vs Predicted Values

![Actual VS Predicted values](Figures/actual_vs_predicted.png)

**Key Observations**

* The optimized XGBoost model achieved an **R² score of 0.715** on the test set.
* Predicted zT values exhibit a strong positive correlation with the experimental values.
* Most samples lie close to the ideal prediction line (*y = x*), indicating good predictive performance.
* The model successfully captures the overall relationship between material descriptors, temperature, and thermoelectric performance.
* Prediction errors increase slightly for high-zT materials, suggesting greater complexity in modeling highly efficient thermoelectric compounds.
* The results demonstrate that the model can effectively serve as a screening tool for identifying promising thermoelectric materials.

**Performance Summary**

| Metric              | Value                   |
| ------------------- | ----------------------- |
| Test R² Score       | 0.715                   |
| Train R² Score      | 0.942                   |
| Model               | XGBoost Regressor       |
| Validation Strategy | 5-Fold Cross Validation |
| CV Mean R²          | 0.752                   |
| CV Std              | 0.066                   |


The close alignment between predicted and actual values confirms the model's ability to learn meaningful structure-property relationships from composition-based and temperature-dependent descriptors.


Output:

```text
final_xgb_top50.pkl

```

---


## Future Work

* Increase dataset size
* Predict ZT for unseen thermoelectric compounds
* Integrate Materials Project data



