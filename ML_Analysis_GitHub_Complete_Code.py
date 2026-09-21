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
# 1. PACKAGES TO INSTALL
# ============================================================

%pip install numpy pandas matplotlib seaborn scikit-learn yellowbrick shap model-diagnostics skimpy deepchecks


# ============================================================
# 2. LIBRARIES
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
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
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

import shap

from model_diagnostics.calibration import plot_reliability_diagram

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity

sns.set_style("whitegrid")


# ============================================================
# GENERAL PREPROCESSING PRINCIPLE
# ============================================================

# Split first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Fit preprocessing only on X_train
# Apply the same fitted preprocessing to X_test


# ============================================================
# 3. DATASET 1 — FuelConsumptionCo2.csv
# Regression: Predict CO2EMISSIONS
# ============================================================

# STEP 3 — Import Dataset

df = pd.read_csv("FuelConsumptionCo2.csv")

print(df.shape)
df.head()


# STEP 4 — Data Exploration

df.head()
df.tail()
df.shape
df.info()
df.describe()
df.isnull().sum()
skim(df)

print("Duplicates:", df.duplicated().sum())
print(df.select_dtypes("object").nunique())
print(df["CO2EMISSIONS"].describe())

num_cols = df.select_dtypes(include=np.number).columns

df[num_cols].boxplot(figsize=(10, 5), rot=45)
plt.title("Outlier Investigation")
plt.show()

sns.histplot(df["CO2EMISSIONS"], kde=True)
plt.title("Target Distribution — CO2EMISSIONS")
plt.show()

sns.scatterplot(data=df, x="ENGINESIZE", y="CO2EMISSIONS")
plt.title("CO2EMISSIONS vs ENGINESIZE")
plt.show()

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation — Numerical Features")
plt.show()


# STEP 5 — Initial Data Validation

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity

check = data_integrity()

check.run(
    Dataset(df, label="CO2EMISSIONS")
).show()


# STEP 6–7 — Split and Preprocess

X = df[
    [
        "ENGINESIZE",
        "CYLINDERS",
        "FUELCONSUMPTION_CITY",
        "FUELCONSUMPTION_HWY",
        "FUELCONSUMPTION_COMB"
    ]
]

y = df["CO2EMISSIONS"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# STEP 8 — Models

lr = LinearRegression()
dt = DecisionTreeRegressor(max_depth=5, random_state=42)
knn = KNeighborsRegressor(n_neighbors=5)
svr = SVR()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
knn.fit(X_train, y_train)
svr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)
pred_dt = dt.predict(X_test)
pred_knn = knn.predict(X_test)
pred_svr = svr.predict(X_test)


# STEP 9–10 — Output and Evaluation

print("Linear Regression")
print("R2:", r2_score(y_test, pred_lr))
print("MAE:", mean_absolute_error(y_test, pred_lr))
print("RMSE:", mean_squared_error(y_test, pred_lr) ** 0.5)


# STEP 11–12 — Diagnostics and Learning Curve

visualizer = ResidualsPlot(LinearRegression())

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

visualizer = PredictionError(LinearRegression())

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

model = LinearRegression().fit(X_train, y_train)

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

shap.plots.beeswarm(shap_values)

LearningCurve(
    KNeighborsRegressor(n_neighbors=5),
    scoring="r2"
).fit(X_train, y_train).show()


# ============================================================
# 4. DATASET 2 — drug200.csv
# Multiclass Classification: Predict Drug
# ============================================================

# STEP 3–4 — Import and Explore

df = pd.read_csv("drug200.csv")

df.head()
df.tail()
df.shape
df.info()
df.describe()
df.isnull().sum()
skim(df)

print("Duplicates:", df.duplicated().sum())
print(df["Drug"].value_counts())
print(df.nunique())

df[["Age", "Na_to_K"]].boxplot(figsize=(7, 4))
plt.title("Outlier Investigation")
plt.show()

sns.countplot(data=df, x="Drug")
plt.title("Target Distribution — Drug")
plt.show()

sns.boxplot(data=df, x="Drug", y="Na_to_K")
plt.title("Na_to_K vs Drug")
plt.show()

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm"
)
plt.show()


# STEP 6–8 — Split, Preprocess and Model

X = df.drop(columns="Drug")
y = df["Drug"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

num_cols = X.select_dtypes(exclude="object").columns
cat_cols = X.select_dtypes(include="object").columns

preprocess = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        num_cols
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        cat_cols
    )
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(
        kernel="linear",
        probability=True,
        random_state=42
    )
}

for name, model in models.items():

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    print(name, pipe.score(X_test, y_test))


# STEP 9–10 — Model Evaluation

pred = pipe.predict(X_test)

print(classification_report(y_test, pred))

ConfusionMatrix(pipe).fit(
    X_train,
    y_train
).score(
    X_test,
    y_test
).show()

ROCAUC(
    pipe,
    classes=sorted(y.unique())
).fit(
    X_train,
    y_train
).score(
    X_test,
    y_test
).show()

PrecisionRecallCurve(
    pipe,
    per_class=True
).fit(
    X_train,
    y_train
).score(
    X_test,
    y_test
).show()


# STEP 11–12 — Diagnostics

ClassificationReport(pipe).fit(
    X_train,
    y_train
).score(
    X_test,
    y_test
).show()


# SHAP is most straightforward after fitting a tree model

tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

Xtr = preprocess.fit_transform(X_train)
Xte = preprocess.transform(X_test)

tree.fit(Xtr, y_train)

explainer = shap.Explainer(tree, Xtr)
shap_values = explainer(Xte)

shap.plots.beeswarm(shap_values)

LearningCurve(
    DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    scoring="accuracy"
).fit(
    Xtr,
    y_train
).show()


# ============================================================
# 5. DATASET 3 — teleCust1000t.csv
# Multiclass Classification: Predict custcat
# ============================================================

df = pd.read_csv("teleCust1000t.csv")

df.head()
df.tail()
df.shape
df.info()
df.describe()
df.isnull().sum()
skim(df)

print("Duplicates:", df.duplicated().sum())
print(df["custcat"].value_counts())

df.drop(columns="custcat").boxplot(
    figsize=(10, 5),
    rot=45
)
plt.title("Outlier Investigation")
plt.show()

sns.countplot(data=df, x="custcat")
plt.title("Target Distribution — custcat")
plt.show()

sns.boxplot(
    data=df,
    x="custcat",
    y="income"
)
plt.title("Income vs custcat")
plt.show()

sns.heatmap(
    df.corr(),
    cmap="coolwarm",
    annot=False
)
plt.title("Correlation")
plt.show()


# Split and Preprocess

X = df.drop(columns="custcat")
y = df["custcat"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Models

logr = LogisticRegression(max_iter=1000)

dt = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

knn = KNeighborsClassifier(n_neighbors=9)

svm = SVC(
    kernel="linear",
    probability=True,
    random_state=42
)

for model in [logr, dt, knn, svm]:

    model.fit(X_train, y_train)

    print(
        model.__class__.__name__,
        model.score(X_test, y_test)
    )


# ============================================================
# 6. DATASET 4 — Obesity_level_prediction_dataset(1).csv
# Multiclass Classification: Predict NObeyesdad
# ============================================================

df = pd.read_csv(
    "Obesity_level_prediction_dataset(1).csv"
)

df.head()
df.tail()
df.shape
df.info()
df.describe()
df.isnull().sum()
skim(df)

print("Duplicates:", df.duplicated().sum())
print(df["NObeyesdad"].value_counts())

df.select_dtypes(include=np.number).boxplot(
    figsize=(10, 5),
    rot=45
)
plt.title("Outlier Investigation")
plt.show()

sns.countplot(
    data=df,
    x="NObeyesdad"
)
plt.xticks(rotation=45)
plt.title("Target Distribution")
plt.show()

sns.boxplot(
    data=df,
    x="NObeyesdad",
    y="Weight"
)
plt.xticks(rotation=45)
plt.title("Weight vs Obesity Level")
plt.show()

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    cmap="coolwarm",
    annot=True,
    fmt=".2f"
)
plt.title("Numerical Correlation")
plt.show()


# Remove duplicates

df = df.drop_duplicates()

X = df.drop(columns="NObeyesdad")
y = df["NObeyesdad"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Preprocessing

num_cols = X.select_dtypes(exclude="object").columns
cat_cols = X.select_dtypes(include="object").columns

preprocess = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        num_cols
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        cat_cols
    )
])


# Models

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "SVM": SVC(
        kernel="linear",
        probability=True,
        random_state=42
    )
}

for name, model in models.items():

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)

    print(name)

    print(
        "Accuracy:",
        accuracy_score(y_test, pred)
    )

    print(
        "Macro F1:",
        f1_score(
            y_test,
            pred,
            average="macro"
        )
    )


# ============================================================
# 7. DATASET 5 — creditcard(in) (1)(2).csv
# Binary Classification: Fraud Detection
# ============================================================

df = pd.read_csv(
    "creditcard(in) (1)(2).csv"
)

df.head()
df.tail()
df.shape
df.info()
df.describe()
df.isnull().sum()
skim(df)

print("Duplicates:", df.duplicated().sum())
print(df["Class"].value_counts())

sns.countplot(
    data=df,
    x="Class"
)
plt.title("Fraud vs Non-Fraud")
plt.show()

sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)

plt.ylim(
    0,
    df["Amount"].quantile(0.99)
)

plt.title("Transaction Amount by Class")
plt.show()

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()


# Remove duplicates

df = df.drop_duplicates()

X = df.drop(columns="Class")
y = df["Class"]


# Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Scale

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Models

logr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

dt = DecisionTreeClassifier(
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)

svm = LinearSVC(
    class_weight="balanced",
    random_state=42
)

logr.fit(X_train, y_train)
dt.fit(X_train, y_train)
svm.fit(X_train, y_train)


# Evaluation

pred = svm.predict(X_test)

print(
    classification_report(
        y_test,
        pred
    )
)

ConfusionMatrix(svm).fit(
    X_train,
    y_train
).score(
    X_test,
    y_test
).show()

# For a binary model with probability/score output:
# use ROC-AUC and Precision-Recall curves.


# ============================================================
# 8. DATASET 6 — customers_raw.csv + purchases.csv
# Cleaning + Regression + K-Means Clustering
# ============================================================

# STEP 3–4 — Import, Clean and Explore

customers = pd.read_csv("customers_raw.csv")
purchases = pd.read_csv("purchases.csv")

customers.head()
customers.tail()
customers.shape
customers.info()
customers.describe()
customers.isnull().sum()
skim(customers)

print(
    "Duplicates:",
    customers.duplicated().sum()
)

print(
    customers.nunique()
)

purchases.head()
purchases.info()


# Cleaning

customers["Name"] = customers["Name"].fillna("Unknown")

customers["Name"] = (
    customers["Name"]
    .str.strip()
    .str.title()
)

customers["City"] = (
    customers["City"]
    .str.strip()
    .str.title()
)

customers = customers.drop_duplicates()

print(customers["City"].unique())
print(customers["Gender"].unique())


# Missing values

customers["Age"] = customers["Age"].fillna(
    customers["Age"].median()
)

customers["Salary"] = customers["Salary"].fillna(
    customers["Salary"].median()
)

# Outlier treatment after the split should be used for a predictive model.
# For this tiny teaching dataset, inspect the suspicious values first.


# Merge

df = customers.merge(
    purchases,
    on="CustomerID",
    how="left"
)

df["TotalPurchases"] = df["TotalPurchases"].fillna(
    df["TotalPurchases"].median()
)

df["MemberSince"] = df["MemberSince"].fillna(
    df["MemberSince"].median()
)

print(df)


# ------------------------------------------------------------
# Regression Model
# ------------------------------------------------------------

X = df[
    [
        "Age",
        "City",
        "Gender",
        "TotalPurchases",
        "MemberSince"
    ]
]

y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

num_cols = [
    "Age",
    "TotalPurchases",
    "MemberSince"
]

cat_cols = [
    "City",
    "Gender"
]

preprocess = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        num_cols
    ),
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        cat_cols
    )
])

lr = Pipeline([
    ("preprocess", preprocess),
    ("model", LinearRegression())
])

lr.fit(X_train, y_train)

pred = lr.predict(X_test)

print(
    "R2:",
    r2_score(y_test, pred)
)

print(
    "MAE:",
    mean_absolute_error(y_test, pred)
)

print(
    "RMSE:",
    mean_squared_error(y_test, pred) ** 0.5
)


# ------------------------------------------------------------
# K-Means Clustering
# ------------------------------------------------------------

Z = preprocess.fit_transform(
    df[
        [
            "Age",
            "City",
            "Gender",
            "TotalPurchases",
            "MemberSince"
        ]
    ]
)

for k in range(2, 6):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = km.fit_predict(Z)

    print(
        k,
        silhouette_score(Z, labels)
    )

km = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

df["Cluster"] = km.fit_predict(Z)


# ============================================================
# 9. MODEL DIAGNOSTICS
# ============================================================

# Regression diagnostics

visualizer = ResidualsPlot(LinearRegression())

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

visualizer = PredictionError(LinearRegression())

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()


# Classification diagnostics

ClassificationReport(pipe).fit(
    X_train,
    y_train
).score(
    X_test,
    y_test
).show()

ConfusionMatrix(pipe).fit(
    X_train,
    y_train
).score(
    X_test,
    y_test
).show()

ROCAUC(pipe).fit(
    X_train,
    y_train
).score(
    X_test,
    y_test
).show()

PrecisionRecallCurve(pipe).fit(
    X_train,
    y_train
).score(
    X_test,
    y_test
).show()


# ============================================================
# 10. SHAP
# ============================================================

explainer = shap.Explainer(
    model,
    X_train
)

shap_values = explainer(X_test)

shap.plots.beeswarm(shap_values)


# ============================================================
# 11. MODEL DIAGNOSTICS — RELIABILITY DIAGRAM
# ============================================================

# Use after obtaining y_test and a probability/score prediction

plot_reliability_diagram(
    y_test,
    prediction_score
)


# ============================================================
# 12. LEARNING CURVES
# ============================================================

LearningCurve(
    model,
    scoring="accuracy"  # use "r2" for regression
).fit(
    X_train,
    y_train
).show()


# ============================================================
# END OF CODE COLLECTION
# ============================================================
