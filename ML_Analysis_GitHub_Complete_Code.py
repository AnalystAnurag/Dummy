# ============================================================
# MACHINE LEARNING ANALYSIS
# Complete Code Collection
# Based on: ML_Analysis_Report_Revised_Simple_Workflow
# ============================================================
#
# Sequence:
# 1. Packages to install
# 2. Libraries
# 3. All dataset / analysis codes
#
# ============================================================
# Step 1. PACKAGES TO INSTALL
# ============================================================

pip install numpy pandas matplotlib seaborn scikit-learn yellowbrick shap model-diagnostics skimpy deepchecks

pip install --upgrade deepchecks


# ============================================================
# Step 2. LIBRARIES
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimpy import skim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    silhouette_score
)

from yellowbrick.model_selection import LearningCurve
from yellowbrick.regressor import ResidualsPlot, PredictionError
from yellowbrick.classifier import (
    ClassificationReport,
    ConfusionMatrix,
    ROCAUC,
    PrecisionRecallCurve
)

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, full_suite

import shap

from model_diagnostics.calibration import plot_reliability_diagram

sns.set_style("whitegrid")

# ============================================================
# Step 3: Import Dataset 
# ============================================================

df = pd.read_csv(DATA_PATH)

df.head()


# ============================================================
# 4. Step 4 — Data Exploration
# ============================================================

# STEP 3 — Import Dataset

df = pd.read_csv("FuelConsumptionCo2.csv")

print(df.shape)
df.head()


# STEP 4 — Data Exploration

## First and last records

df.head()
df.tail()

## Shape of data

df.shape

## Information

df.info()

## Descriptive statistics

df.describe()

## Missing value check

df.isnull().sum().sort_values(ascending=False)
(df.isnull().mean() * 100).sort_values(ascending=False)

## Using Skimpy for better initial exploration

skim(df)

## Duplicate records

print("Duplicates:", df.duplicated().sum())

## To remove duplicates

df = df.drop_duplicates()

## Unique values

df.nunique().sort_values()

cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    print(f"\n{col}")
    print(df[col].unique())

## Separate numerical and categorical variable 

X = df.drop(columns=TARGET)
y = df[TARGET]

numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

print("Numeric:", numeric_cols)
print("Categorical:", categorical_cols)

## Outlier check

for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Outlier Investigation - {col}")
    plt.show()

or 

IQR based:
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    count = ((df[col] < lower) | (df[col] > upper)).sum()
    
    print(f"{col}: {count} potential outliers")

## Target Variable distribution 

for numerical target variable:

sns.histplot(df[TARGET], kde=True)
plt.title("Target Distribution")
plt.show()

for categorical target variable: 
 
sns.countplot(x=df[TARGET])
plt.title("Target Distribution")
plt.show()

## Target vs Numeric features

for col in numeric_cols:
    if col != TARGET:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[col], y=df[TARGET])
        plt.title(f"{TARGET} vs {col}")
        plt.show()

## Correlation plot of numerical features

corr = df.select_dtypes(include=np.number).corr()

plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# ============================================================
# Step 5 — Data Validation
# ============================================================

import numpy as np

np.Inf = np.inf

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity

check = data_integrity()

check.run(
    Dataset(obesity, label="NObeyesdad")
).show()

# ============================================================
# Step 6 — Train-Test Split
# ============================================================

X = df.drop(columns=TARGET)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
stratify = y
)

# ============================================================
# Step 7 — Data Pre-processing
# ============================================================


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder
)

# Identify columns
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

# Define special categorical columns
binary_cols = []
ordinal_cols = []

# Remaining nominal columns
nominal_cols = [
    col for col in categorical_cols
    if col not in binary_cols + ordinal_cols
]

# Numerical pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Binary pipeline
binary_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder())
])

# Ordinal pipeline
ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

# Nominal pipeline
nominal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("binary", binary_pipeline, binary_cols),
    ("ordinal", ordinal_pipeline, ordinal_cols),
    ("nominal", nominal_pipeline, nominal_cols)
])

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

# ============================================================
# Step 7 — Training the model
# ============================================================

## Linear Regression: 

### Training the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

model.fit(X_train, y_train)

### Model Output Summary
print(model)

### Predicting the model

y_pred = model.predict(X_test)

pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
}).head(10)

### Model Evaluation

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("R²   :", round(r2, 4))
print("MAE  :", round(mae, 4))
print("MSE  :", round(mse, 4))
print("RMSE :", round(rmse, 4))

### Model Diagnostic

#### Residual Plot

visualizer = ResidualsPlot(model)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

#### Prediction Error 

visualizer = PredictionError(model)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

#### SHAP Importance

X_train_transformed = model.named_steps["preprocessor"].transform(X_train)
X_test_transformed = model.named_steps["preprocessor"].transform(X_test)

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

X_train_shap = pd.DataFrame(
    X_train_transformed,
    columns=feature_names
)

X_test_shap = pd.DataFrame(
    X_test_transformed,
    columns=feature_names
)

explainer = shap.LinearExplainer(
    model.named_steps["model"],
    X_train_shap
)

shap_values = explainer(X_test_shap)

shap.summary_plot(shap_values, X_test_shap)

### Learning Curve

visualizer = LearningCurve(
    model,
    scoring="r2",
    cv=5
)

visualizer.fit(X, y)
visualizer.show()


##################
## Logistic Regression

### Training the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

### Model output

print(model)

y_pred = model.predict(X_test)

pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
}).head(10)

### Classification metrics
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, y_pred, average="weighted"))

complete report 

print(classification_report(y_test, y_pred))

### Model Diagnostics

visualizer = ClassificationReport(model)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

### ROC-AUC Curve
visualizer = ROCAUC(model)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

### Precision - Recall 

visualizer = PrecisionRecallCurve(model)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

### Model Diagnostic

X_train_transformed = model.named_steps["preprocessor"].transform(X_train)
X_test_transformed = model.named_steps["preprocessor"].transform(X_test)

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

X_train_shap = pd.DataFrame(
    X_train_transformed,
    columns=feature_names
)

X_test_shap = pd.DataFrame(
    X_test_transformed,
    columns=feature_names
)

explainer = shap.LinearExplainer(
    model.named_steps["model"],
    X_train_shap
)

shap_values = explainer(X_test_shap)

shap.summary_plot(shap_values, X_test_shap)

### Learning Curve

visualizer = LearningCurve(
    model,
    scoring="f1_weighted",
    cv=5
)

visualizer.fit(X, y)
visualizer.show()

###################
###Decision Tree

############Decision Tree Regression############

### Training the Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ))
])

model.fit(X_train, y_train)

### Model output & Evaluation
print(model) 
y_pred = model.predict(X_test)

pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
}).head(10)

### Model Evaluation
print("R²   :", r2_score(y_test, y_pred))
print("MAE  :", mean_absolute_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))

### Model Diagnostics

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = model.named_steps[
    "model"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(
    "Importance",
    ascending=False
)

importance_df.head(10)

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top Feature Importances")
plt.show()


### SHAP
X_test_transformed = model.named_steps["preprocessor"].transform(X_test)

explainer = shap.TreeExplainer(
    model.named_steps["model"]
)

shap_values = explainer.shap_values(X_test_transformed)

shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=feature_names
)

### Learning curve
visualizer = LearningCurve(
    model,
    scoring="r2",
    cv=5
)

visualizer.fit(X, y)
visualizer.show()

#####Decisiontree Classification

### Training the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ))
])

model.fit(X_train, y_train)

### Prediction
y_pred = model.predict(X_test)

### Model evaluation

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
print("F1       :", f1_score(y_test, y_pred, average="weighted"))

### Confusion Matrix

visualizer = ConfusionMatrix(model)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

### Learning curve

visualizer = LearningCurve(
    model,
    scoring="f1_weighted",
    cv=5
)

visualizer.fit(X, y)
visualizer.show()

# --- Additional Visualization for Decision Tree ---

from sklearn import tree
import matplotlib.pyplot as plt

# Decision Tree Diagram
plt.figure(figsize=(20, 10))

tree.plot_tree(
    dt_model,
    feature_names=list(X.columns),
    class_names=[str(c) for c in sorted(y.unique())]
                 if hasattr(dt_model, "classes_") else None,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree - Tree Diagram", fontsize=16)
plt.show()


# Feature Importance Plot
plt.figure(figsize=(10, 5))

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

tree.plot_tree(
    dt_model,
    feature_names=feature_names,
    class_names=[str(c) for c in sorted(y.unique())]
                 if hasattr(dt_model, "classes_") else None,
    filled=True,
    rounded=True,
    fontsize=10
)


######################
####Support Vector Machine

###Support Vector Machine Regression 
#### Training the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", SVR(kernel="rbf"))
])

model.fit(X_train, y_train)

#### Predicting

y_pred = model.predict(X_test)

pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
}).head(10)

#### Model evaluation
print("R²   :", r2_score(y_test, y_pred))
print("MAE  :", mean_absolute_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))

#### Model diagnostic

##### Prediction error

visualizer = PredictionError(model)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

##### Residuals plot

visualizer = ResidualsPlot(model)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

#### learning curve

visualizer = LearningCurve(
    model,
    scoring="r2",
    cv=5
)

visualizer.fit(X, y)
visualizer.show()

#####Support vector machine classifier

### Training the model 

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ))
])

model.fit(X_train, y_train)

#### predicting

y_pred = model.predict(X_test)

#### Model evaluation

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
print("F1       :", f1_score(y_test, y_pred, average="weighted"))

### Classification report

visualizer = ClassificationReport(model)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

### Confusion Matrix 

visualizer = ConfusionMatrix(model)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

#### Learning curve

visualizer = LearningCurve(
    model,
    scoring="f1_weighted",
    cv=5
)

visualizer.fit(X, y)
visualizer.show()

##################
#### K-Means Clustering(No Train Test split as Unsupervised)

### Data Import

df = pd.read_csv("your_dataset.csv")

### Data Exploration

df.head()
df.tail()
df.shape
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
skim(df)
df.nunique()

### correlation plot
sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.show()

### Deepchecks data validation

ds = Dataset(
    df,
    cat_features=df.select_dtypes(
        exclude=np.number
    ).columns.tolist()
)

result = data_integrity().run(ds)

result.show()

### X = df.copy()

X = X.drop(columns=["Customer_ID"], errors="ignore") ## incase any identifiers

### Preprocessing

numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

### transforming
X_processed = preprocessor.fit_transform(X)

### Training model

inertia = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    
    kmeans.fit(X_processed)
    inertia.append(kmeans.inertia_)

### Elbow Method
plt.plot(range(2, 11), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

### put the value of k from elbow method
kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_processed)

### cluster output
df["Cluster"] = clusters

df["Cluster"].value_counts().sort_index()

### cluster sizes

sns.countplot(x=df["Cluster"])

plt.title("Cluster Distribution")
plt.show()

### clustering evaluation
silhouette = silhouette_score(
    X_processed,
    clusters
)

print("Silhouette Score:", round(silhouette, 4))

### Cluster Profiling

cluster_profile = df.groupby("Cluster").mean(
    numeric_only=True
)

cluster_profile

####visually plotting

cluster_profile.T.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Cluster Profile")
plt.ylabel("Average Value")
plt.show()

# --- Additional Visualization for K-Means ---

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Reduce data to 2 dimensions
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_processed)

# Get cluster predictions
kmeans_pred = kmeans.predict(X_processed)

# Plot clusters
plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=kmeans_pred,
    palette="tab10",
    s=60
)

# Transform cluster centroids into PCA space
centroids_pca = pca.transform(
    kmeans.cluster_centers_
)

plt.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    c="black",
    marker="X",
    s=200,
    label="Centroids"
)

plt.title("K-Means Clustering (PCA Projection)", fontsize=14)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()


# Cluster counts
print("Cluster counts:")

print(
    pd.Series(kmeans_pred).value_counts().sort_index()
)


####################
###KNN

#####KNN Regression
### Training the model

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", KNeighborsRegressor(
        n_neighbors=5
    ))
])

model.fit(X_train, y_train)

#### predicting
y_pred = model.predict(X_test)

pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
}).head(10)

#### Evaluating model
print("R²   :", r2_score(y_test, y_pred))
print("MAE  :", mean_absolute_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))

#### Model Diagnostic

###Prediction error
visualizer = PredictionError(model)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

### Learning curve
visualizer = LearningCurve(
    model,
    scoring="r2",
    cv=5
)

visualizer.fit(X, y)
visualizer.show()


#############KNN Classification
### traing the model

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", KNeighborsClassifier(
        n_neighbors=5
    ))
])

model.fit(X_train, y_train)

### predicting 

y_pred = model.predict(X_test)

### model evaluation

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
print("F1       :", f1_score(y_test, y_pred, average="weighted"))

#### classification report

visualizer = ClassificationReport(model)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

### confusion matrix
visualizer = ConfusionMatrix(model)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()


#### Learning curve

visualizer = LearningCurve(
    model,
    scoring="f1_weighted",
    cv=5
)

visualizer.fit(X, y)
visualizer.show()

# --- KNN Decision Boundary Visualization ---

from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# Transform the original data using fitted preprocessing
X_transformed = model.named_steps["preprocessor"].transform(X_train)

# Use first two transformed features
X_vis = X_transformed[:, :2]

# Train a small visualization KNN using the two selected features
knn_vis = KNeighborsClassifier(n_neighbors=5)

knn_vis.fit(X_vis, y_train)

# Create mesh grid
h = 0.02

x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

# Predict grid
Z = knn_vis.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3,
    cmap="coolwarm"
)

sns.scatterplot(
    x=X_vis[:, 0],
    y=X_vis[:, 1],
    hue=y_train,
    edgecolor="black",
    s=60
)

plt.title(
    "KNN Classification - Decision Boundary",
    fontsize=14
)

plt.xlabel("Feature 1 (transformed)")
plt.ylabel("Feature 2 (transformed)")

plt.legend(title="Class")
plt.show()






# ============================================================
# OPTIONAL — MANUAL ENCODING
# ============================================================

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


# ------------------------------------------------------------
# 1. LABEL ENCODING
# Use mainly for binary/multiclass target
# ------------------------------------------------------------

le = LabelEncoder()

df["Target_Encoded"] = le.fit_transform(df[TARGET])

print("Classes:", le.classes_)


# ------------------------------------------------------------
# 2. ONE-HOT ENCODING — pandas
# Use for nominal categorical variables
# ------------------------------------------------------------

df_onehot = pd.get_dummies(
    df,
    columns=["Category"],
    dtype=int
)


# ------------------------------------------------------------
# 3. ONE-HOT ENCODING — sklearn
# ------------------------------------------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded = encoder.fit_transform(
    df[["Category"]]
)

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(["Category"])
)

print(encoded_df.head())


# ------------------------------------------------------------
# 4. ORDINAL ENCODING — sklearn
# Use when categories have a meaningful order
# ------------------------------------------------------------

ordinal_encoder = OrdinalEncoder(
    categories=[
        ["Low", "Medium", "High"]
    ]
)

df["Level_Encoded"] = ordinal_encoder.fit_transform(
    df[["Level"]]
)


# ------------------------------------------------------------
# 5. ORDINAL ENCODING — manual mapping
# ------------------------------------------------------------

level_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

df["Level_Encoded"] = df["Level"].map(level_map)

####### Outlier Treatment#######
for col in num_cols:

    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    X_train[col] = X_train[col].clip(lower_bound, upper_bound)
    X_test[col] = X_test[col].clip(lower_bound, upper_bound)

ML MODEL INTERPRETATION CHEAT SHEET
====================================

Use the same 5-part framework for almost every ML model:

1. DATA VALIDATION / DEEPCHECKS
2. MODEL OUTPUT SUMMARY
3. MODEL EVALUATION
4. MODEL DIAGNOSTICS
5. LEARNING CURVE


============================================================
1. DATA VALIDATION / DEEPCHECKS
============================================================

Main question:
"Can I trust my data and the relationship between train and test?"

Typical checks:

- Missing values
  High percentage of nulls = data quality issue.

- Duplicate rows
  Duplicates may indicate repeated observations and should be investigated.

- Mixed data types
  A column containing inconsistent types requires data cleaning.

- Single-value columns
  A column containing only one unique value usually provides little/no predictive information.

- Feature correlation
  Very high correlation may indicate redundancy or multicollinearity.

- Feature-label correlation
  Shows whether features have a relationship with the target. A relationship does not imply causality.

- Train/Test drift
  Major distribution differences between train and test can affect generalization.

- New categories
  Categories appearing in test but not train can cause encoding problems.

- Outliers
  Extreme observations should be investigated before deciding whether to cap, transform, or retain them.

- Label drift
  A substantial difference in target distribution between train and test may indicate population differences.

Good interpretation:
"The data validation checks indicate that the dataset is structurally suitable for modeling, with no major issues related to missing values, duplicates, data types, or train-test distribution differences."

If an issue exists:
"The validation identified a potential issue with feature distribution between the training and test datasets. This may affect model generalization and should be investigated before interpreting test performance."

Professional wording:
Do not automatically say "The data is good."
Prefer:
"No major data-quality issue was detected in the checks performed."


============================================================
2. MODEL OUTPUT SUMMARY
============================================================

A. LINEAR REGRESSION
--------------------

Look at:
- Coefficients
- Intercept
- R-squared
- Adjusted R-squared
- p-values
- F-statistic

Coefficient interpretation:
If Income coefficient = 2.5:

"Holding other variables constant, a one-unit increase in Income is associated with an estimated 2.5-unit increase in the target."

Negative coefficient:
"The variable has a negative estimated association with the target."

Avoid automatically saying "causes."


B. LOGISTIC REGRESSION
----------------------

Look at:
- Coefficients
- Odds ratios
- p-values
- Statistical significance
- Model fit

Odds ratio:
OR = exp(coefficient)

Example:
Coefficient = 0.40
OR = exp(0.40) approximately 1.49

Interpretation:
"A one-unit increase in Income is associated with approximately 49% higher odds of the positive class, holding other variables constant."

Negative coefficient:
"The variable is associated with lower odds of the positive class."


C. DECISION TREE
----------------

Look at:
- Tree depth
- Splitting variables
- Feature importance
- Leaf structure

Interpretation:
"The tree primarily splits observations based on Age and Income, indicating that these variables contribute strongly to the model's decision structure."

Do not discuss regression-style coefficients because decision trees do not have conventional regression coefficients.



E. KNN
------

Look at:
- k (number of neighbours)
- Distance metric
- Training performance
- Test performance

Interpretation:
"The model uses the nearest k observations to determine the prediction. The choice of k influences the bias-variance trade-off."

Small k:
- More flexible
- More sensitive to noise
- Higher variance

Large k:
- Smoother decision boundary
- Less sensitive to individual observations
- Potentially higher bias


F. SVM
------

Look at:
- Kernel
- C
- Gamma
- Support vectors
- Performance

Interpretation:
"The SVM separates observations by constructing a decision boundary that maximizes the margin between classes."

C:
Controls the penalty for classification errors.

Gamma:
Controls the influence of individual observations for applicable kernels.


G. K-MEANS
---------

Look at:
- Number of clusters (k)
- Centroids
- Cluster sizes
- Within-cluster variation
- Silhouette score

Interpretation:
"The clustering divides observations into groups based on similarity in the selected features."


============================================================
3. MODEL EVALUATION
============================================================

A. REGRESSION
-------------

Main metrics:
- R-squared
- MAE
- RMSE

R-SQUARED:
R² = 1 - (SSres / SStot)

Example:
R² = 0.82

Interpretation:
"The model explains approximately 82% of the variation in the dependent variable."

Higher R² is generally better, but it should not be interpreted alone.


MAE:
Example:
MAE = 5.2

Interpretation:
"On average, predictions differ from actual values by approximately 5.2 units."

MAE is in the same unit as the target.


RMSE:
Example:
RMSE = 8.7

Interpretation:
"The RMSE indicates an average prediction error magnitude of approximately 8.7 units, with greater sensitivity to larger errors."

Important:
Do NOT say:
"RMSE is 8.7%, so accuracy is 91.3%."

RMSE is in the target variable's units.


B. CLASSIFICATION
-----------------

Main metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC


ACCURACY:
Accuracy = (TP + TN) / Total

Example:
Accuracy = 0.89

Interpretation:
"The model correctly classifies approximately 89% of observations."

Caution:
Accuracy alone can be misleading with imbalanced classes.


PRECISION:

Question:
"Of the observations predicted as positive, how many were actually positive?"

High precision:
Fewer false positives.


RECALL:

Question:
"Of the actual positive observations, how many did the model identify?"

High recall:
Fewer false negatives.


F1-SCORE:

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Interpretation:
"F1-score provides a balance between precision and recall."

Higher F1 is generally better.


ROC-AUC:

Approximate interpretation:
0.5 = roughly random discrimination
1.0 = perfect discrimination

Example:
AUC = 0.87

Interpretation:
"The model demonstrates good discrimination between the positive and negative classes."

Do NOT say:
"AUC = 0.87 means 87% accuracy."

AUC measures discrimination, not accuracy.


C. CLUSTERING
-------------

Main metrics:
- Silhouette score
- Inertia

SILHOUETTE SCORE:
Higher values generally indicate better-defined clusters.

Interpretation:
"A higher silhouette score indicates that observations are relatively well separated from other clusters and relatively similar to observations within their own cluster."

INERTIA:
Lower inertia means observations are closer to their cluster centroids.

Important:
Inertia generally decreases as k increases, so it should not be interpreted alone. Use the elbow method to help select k.


============================================================
4. CONFUSION MATRIX
============================================================

Structure:

                 Predicted
                Negative Positive

Actual Negative    TN       FP
Actual Positive    FN       TP


TN = True Negative
FP = False Positive
FN = False Negative
TP = True Positive


FALSE POSITIVE:
Model predicts Positive
Reality is Negative

FALSE NEGATIVE:
Model predicts Negative
Reality is Positive


Example:
TN = 850
FP = 40
FN = 60
TP = 250

Interpretation:
"The model correctly classified 850 negative and 250 positive observations. It produced 40 false positives and 60 false negatives."

Then connect the result to the business problem.

Example:
"The relatively higher number of false negatives indicates that some actual positive cases are being missed."


============================================================
5. MODEL DIAGNOSTICS
============================================================

Main question:
"Where and how is the model going wrong?"


A. REGRESSION RESIDUALS PLOT
----------------------------

GOOD:

Residuals are randomly scattered around zero.

Interpretation:
"Residuals are approximately randomly distributed around zero, suggesting that the model captures the systematic relationship reasonably well."

BAD:

Curved or systematic pattern.

Interpretation:
"A systematic pattern is visible in the residuals, suggesting that the model may not adequately capture the underlying relationship."


B. PREDICTION ERROR PLOT
------------------------

Compare:
Actual vs Predicted

GOOD:
Points close to the diagonal line.

Interpretation:
"Predicted values are generally close to actual values, indicating good predictive performance."

BAD:
Points widely scattered from the diagonal.

Interpretation:
"The model exhibits substantial prediction errors, particularly for certain observations."


C. CLASSIFICATION REPORT
-------------------------

Look at:
- Precision
- Recall
- F1-score
- Support

Example:

              Precision   Recall   F1
Class 0          0.92      0.95   0.93
Class 1          0.78      0.70   0.74

Interpretation:
"The model performs better for Class 0 than Class 1. The lower recall for Class 1 indicates that a greater proportion of actual Class 1 observations are being missed."


D. ROC CURVE
------------

Good:
Curve closer to the upper-left corner.

Poor:
Curve close to the diagonal line.

Interpretation:
"The ROC curve indicates how effectively the classifier distinguishes between the two classes across different classification thresholds."


E. PRECISION-RECALL CURVE
-------------------------

Particularly useful for imbalanced datasets.

Interpretation:
"If precision and recall remain relatively high, the model maintains a reasonable balance between identifying positive observations and limiting false positives."


============================================================
6. LEARNING CURVE
============================================================

Main question:
"Is the model underfitting or overfitting?"

The learning curve normally shows:

- Training score
- Validation score

against:

- Number of training examples


CASE 1: GOOD FIT
----------------

Training and validation scores converge at a reasonably high level.

Interpretation:
"The training and validation scores converge to a relatively high level, suggesting that the model generalizes well and does not show substantial overfitting."


CASE 2: OVERFITTING
-------------------

Example:

Training score    = 0.98
Validation score  = 0.75

Large persistent gap.

Interpretation:
"The model shows signs of overfitting because the training score remains substantially higher than the validation score."

Possible solutions:
- Reduce model complexity
- Regularization
- More training data
- Feature selection
- Cross-validation
- Pruning for Decision Tree
- Hyperparameter tuning


CASE 3: UNDERFITTING
--------------------

Example:

Training score    = 0.65
Validation score  = 0.62

Both scores are low and close.

Interpretation:
"The model appears to be underfitting because both training and validation scores remain relatively low, indicating that the model may be too simple to capture the underlying patterns."

Possible solutions:
- Increase model complexity
- Add useful features
- Reduce excessive regularization
- Tune hyperparameters


CASE 4: MORE DATA MAY HELP
--------------------------

If validation performance continues to improve as training size increases:

Interpretation:
"The validation performance continues to improve as the training set increases, suggesting that additional training data may improve generalization."


============================================================
7. QUICK LEARNING CURVE CHEAT SHEET
============================================================

High training + high validation
→ Good fit / good generalization

High training + low validation
→ Overfitting / high variance

Low training + low validation
→ Underfitting / high bias

Validation score improving with more data
→ Additional data may help


============================================================
8. MASTER INTERPRETATION FRAMEWORK
============================================================

For almost every ML model, answer in this order:

1. DATA VALIDATION
   "Is the data trustworthy?"

   Check:
   - Missing values
   - Duplicates
   - Data types
   - Outliers
   - Train-test drift
   - Unexpected categories


2. MODEL OUTPUT
   "What has the model learned?"

   Look at:
   - Coefficients
   - Feature importance
   - Tree structure
   - Clusters
   - Support vectors


3. MODEL EVALUATION
   "How well does it predict?"

   Regression:
   - R²
   - MAE
   - RMSE

   Classification:
   - Accuracy
   - Precision
   - Recall
   - F1
   - ROC-AUC

   Clustering:
   - Silhouette score
   - Inertia


4. MODEL DIAGNOSTICS
   "Where/how does it fail?"

   Look at:
   - Residuals
   - Prediction errors
   - Confusion matrix
   - ROC curve
   - Precision-Recall curve
   - Misclassified observations


5. LEARNING CURVE
   "Is it underfitting or overfitting?"

   High train + high validation
   → Good fit

   High train + low validation
   → Overfitting

   Low train + low validation
   → Underfitting


============================================================
9. UNIVERSAL EXAM INTERPRETATION TEMPLATE
============================================================

"The model's [metric/output] indicates [what happened]. The diagnostic plot shows [pattern]. Therefore, the model appears to [generalization/fit characteristic], with [specific issue if any]."


Examples:

Regression:
"The model achieves an R² of 0.82, indicating that approximately 82% of the variation in the target is explained by the model. The residuals are reasonably scattered around zero, suggesting no major systematic pattern."

Classification:
"The model achieves an accuracy of 89% with an F1-score of 0.84. The confusion matrix indicates that false negatives are higher than false positives, suggesting that the model misses some actual positive cases."

Decision Tree:
"The Decision Tree identifies Income and Age as important splitting variables. The learning curve shows substantially higher training performance than validation performance, indicating potential overfitting."

KNN:
"The KNN model's performance depends on the selected number of neighbours. A small k produces a more flexible model, whereas a larger k produces a smoother decision boundary."

K-Means:
"The clustering solution produces relatively well-separated groups based on the silhouette score. The cluster assignments should be interpreted as groups of similar observations rather than supervised predictions."

