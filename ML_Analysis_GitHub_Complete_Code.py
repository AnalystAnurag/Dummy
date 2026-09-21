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

