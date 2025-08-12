# Breast Cancer Classification with Machine Learning

## Project Description
This project builds a Machine Learning model to classify whether a tumor is malignant or benign based on features from the **Breast Cancer Wisconsin Diagnostic Dataset**. The model applies preprocessing, handles data imbalance with **SMOTE**, performs hyperparameter tuning, and uses **SHAP** for model interpretation.

## Dataset
The dataset used is `breastCancer.csv`, containing features extracted from digitized images of fine needle aspirates (FNA) of breast masses. These features describe characteristics of the cell nuclei.

**Data Source:**  
Breast Cancer Wisconsin (Diagnostic) Dataset, accessed via Kaggle — *"breast-cancer-wisconsin-data"* by UCIML.  
[https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

### Features
- **ID number**
- **Diagnosis**: M = malignant, B = benign
- **30 numeric features** describing cell characteristics (radius, texture, perimeter, area, smoothness, etc.)

## Workflow
1. **Import Libraries**  
   Uses `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `imblearn`, and `shap`.

2. **Data Preprocessing**  
   - Remove irrelevant columns (`Unnamed: 32`)
   - Handle missing values
   - Encode target variable (M/B → 1/0)

3. **Train-Test Split**  
   Split data into training and testing sets using `train_test_split`.

4. **Handling Class Imbalance**  
   Apply **SMOTE** (Synthetic Minority Over-sampling Technique) to balance classes.

5. **Model Training**  
   - Train a classifier (e.g., RandomForest, Decision Tree, etc.)
   - Use `GridSearchCV` for hyperparameter tuning
   - Apply `cross_val_score` for performance validation

6. **Evaluation**  
   - **Accuracy Score**
   - **Confusion Matrix**
   - **Classification Report**
   - **ROC Curve & AUC Score**

7. **Model Interpretation**  
   Use **SHAP (SHapley Additive exPlanations)** to explain model predictions and visualize feature importance.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/username/breast-cancer-classification.git
   cd breast-cancer-classification
   ```
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn imbalanced-learn shap
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook filename.ipynb
   ```
4. Make sure `breastCancer.csv` is in the same directory.

## Project Structure
```
.
├── notebook.ipynb             # Jupyter Notebook with code
└── README.md                  # Project documentation
```

## License
This project is intended for educational purposes. You may use and modify it freely.