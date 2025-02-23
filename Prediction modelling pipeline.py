# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 22:22:35 2025

@author: Dr Bashir Ssuna, MBChB, MSc
Makerere University
"""

# -*- coding: utf-8 -*-
"""
Author: Dr. Bashir Ssuna
Date: 02-Feb-2025
Biostastician
Makerere University Hospital

"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"D:\Youtube channel\outcome_imputed.csv")

data.head()

data.drop(['Unnamed: 0', 'ID'], axis=1, inplace=True)

#%%
"""data structure"""

data.info()

#%%
"""convert categorical variables to categorical and numericals to numeric"""

##A. selecting categorical variables
categorical_vars = [
   "feature22", "feature23", "feature24", "feature25",
  "feature27", "feature28", "feature29", "feature30", "feature31"
]

##B. selecting numeric variables
numerical_vars = ["feature1", "feature2","feature3", "feature4", "feature5",
                   "feature6", "feature7", "feature8","feature9",
                   "feature10", "feature11","feature12", "feature13","feature14", "feature15", "feature16",
                   "feature17", 'feature19',"feature20",
                  "feature18","feature21"
]

#%%
## All variables
all_vars = categorical_vars + numerical_vars

#%%


#%%
X = data[categorical_vars + numerical_vars]
y = data['outcome']

# Convert categorical variables to appropriate data types
data[categorical_vars] = data[categorical_vars].apply(lambda x: x.astype('category'))
data['outcome'] = data['outcome'].astype('category')
data[numerical_vars] = data[numerical_vars].astype('float64')

#%%
# summarize dataset
from numpy import unique
total = len(y)
classes = unique(y)

for c in classes:
    n_examples = len(y[y==c])
    percent = n_examples/total*100
    print('>  class=%d: %d/%d (%.1f%%)'%(c, n_examples, total, percent))

#%%
"""Plot the outcome"""

# binary outcome is already defined as y
y = data['outcome']

# Calculate the distribution
distribution = y.value_counts()
labels = ['No outcome', 'outcome']  # 1 = outcome, 0 = No outcome
colors = ['green', 'red']  # Green for outcome, red for No outcome

# Calculate percentages
total = distribution.sum()
percentages = (distribution / total * 100).round(1)

# Create labels with n (%)
label_texts = [f"{labels[i]}: {distribution[i]} ({percentages[i]}%)" for i in range(len(labels))]

# Plot the pie chart
fig, ax = plt.subplots()
ax.pie(distribution, labels=label_texts, colors=colors, startangle=90, textprops={'fontsize': 10})
ax.set_title("Distribution of outcome", fontsize=14)

# Display the chart
plt.tight_layout()
plt.show()


#%%
"""Plot histograms and normal QQ plots for numerical variables"""

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def plot_histograms_qqplots(data, numerical_vars):
    for var in numerical_vars:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram with KDE
        sns.histplot(data[var], bins=30, kde=True, ax=axes[0])
        axes[0].set_title(f'Histogram & KDE of {var}')
        axes[0].set_xlabel(var)
        axes[0].set_ylabel('Frequency')

        # QQ plot
        stats.probplot(data[var], dist="norm", plot=axes[1])
        axes[1].set_title(f'QQ Plot of {var}')

        plt.tight_layout()
        plt.show()

# Call function with your dataframe and numerical variables list
plot_histograms_qqplots(data, numerical_vars)

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_tiled_histograms(data, numerical_vars, y_var, cols=3):
    num_vars = len(numerical_vars)
    rows = math.ceil(num_vars / cols)  # Compute number of rows dynamically

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))  # Adjust figure size
    axes = axes.flatten()  # Flatten axes to easily index

    for i, var in enumerate(numerical_vars):
        sns.histplot(data, x=var, hue=y_var, bins=30, kde=True, element='step', stat='density', ax=axes[i])
        axes[i].set_title(f'Distribution of {var} by {y_var}')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Density')
        axes[i].legend(title=y_var, labels=[f'{y_var} = 0', f'{y_var} = 1'])
        axes[i].grid(True)

    # Hide any empty subplots if the number of variables isn't a perfect multiple of cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Call function with dataframe, numerical variables, and outcome variable
plot_tiled_histograms(data, numerical_vars, 'outcome')

#%%
"""Apply scaling"""

import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings

# hide warnings
warnings.filterwarnings("ignore")

def plot_tiled_histograms_scaled(data, numerical_vars, y_var, cols=3):
    # Filter numerical_vars to only include those present in data
    numerical_vars = [var for var in numerical_vars if var in data.columns]

    if not numerical_vars:
        print("No valid numerical variables found in the dataset.")
        return

    # Scale numerical variables
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(data[numerical_vars])  # Standardize
    scaled_data = pd.DataFrame(scaled_values, columns=numerical_vars)  # Convert back to DataFrame
    scaled_data[y_var] = data[y_var].values  # Add the outcome variable back

    num_vars = len(numerical_vars)
    rows = math.ceil(num_vars / cols)  # Compute rows dynamically

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))  # Adjust figure size
    axes = axes.flatten()  # Flatten axes for easy indexing

    for i, var in enumerate(numerical_vars):
        sns.histplot(scaled_data, x=var, hue=y_var, bins=30, kde=True, element='step', stat='density', ax=axes[i])
        axes[i].set_title(f'Distribution of Scaled {var} by {y_var}')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Density')
        axes[i].legend(title=y_var)
        axes[i].grid(True)

    # Hide extra empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Call function with dataframe, numerical variables, and outcome variable
plot_tiled_histograms_scaled(data, numerical_vars, 'outcome')

#%%
"""Apply 1+log transformation"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import scipy.stats as stats

def plot_transformed_histograms_qq(data, numerical_vars, y_var, cols=3):
    # Filter numerical_vars to only include those present in data
    numerical_vars = [var for var in numerical_vars if var in data.columns]

    if not numerical_vars:
        print("No valid numerical variables found in the dataset.")
        return

    # Apply 1 + log transformation
    transformed_data = data[numerical_vars].apply(lambda x: np.log1p(x))

    # Add outcome variable for plotting
    transformed_data[y_var] = data[y_var].values

    num_vars = len(numerical_vars)
    rows = math.ceil(num_vars / cols)  # Compute number of rows dynamically

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))  # Adjust figure size
    axes = axes.flatten()

    for i, var in enumerate(numerical_vars):
        sns.histplot(transformed_data, x=var, hue=y_var, bins=30, kde=True, element='step', stat='density', ax=axes[i])
        axes[i].set_title(f'1+log Transformed Distribution of {var}')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Density')
        axes[i].legend(title=y_var)
        axes[i].grid(True)

    # Hide extra empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # Generate QQ Plots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))  # Same layout for QQ plots
    axes = axes.flatten()

    for i, var in enumerate(numerical_vars):
        stats.probplot(transformed_data[var], dist="norm", plot=axes[i])
        axes[i].set_title(f'QQ Plot of 1+log Transformed {var}')

    plt.tight_layout()
    plt.show()

# Call function with dataframe, numerical variables, and outcome variable
plot_transformed_histograms_qq(data, numerical_vars, 'outcome')

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = data[numerical_vars]  # Select only numerical predictors
y = data['outcome'].astype(int)  # Ensure y is integer (binary)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Logistic Regression with LASSO (L1 penalty)
log_lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
log_lasso.fit(X_train_scaled, y_train)

# Predict probabilities and class labels
y_pred_prob = log_lasso.predict_proba(X_test_scaled)[:, 1]  # Probabilities for class 1
y_pred_class = log_lasso.predict(X_test_scaled)  # Binary predictions (0 or 1)

# Compute residuals (difference between actual and predicted probabilities)
residuals = y_test - y_pred_prob

# Print some results
print("First 10 Actual Values:", y_test.values[:10])
print("First 10 Predicted Probabilities:", y_pred_prob[:10])
print("First 10 Residuals:", residuals[:10])

#%%
# Histogram of Residuals
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=30, density=True, alpha=0.6, color='b')
plt.title("Histogram of Residuals (Logistic Regression with LASSO)")
plt.xlabel("Residual Value")
plt.ylabel("Density")
plt.show()

# QQ Plot
plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals (Logistic Regression with LASSO)")
plt.show()

#%%
# Check for multicollinearity and remove highly correlated features
X = data[numerical_vars]
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= 0.69)]

print(f"Highly correlated features to remove: {to_drop}")
print(f"Number of features to remove: {len(to_drop)}")

#%%
# Plot correlation heatmap
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 7})
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title("Correlation Heatmap")
plt.show()

#%%

##A. selecting categorical variables
categorical_vars = [
   "feature22", "feature23", "feature24", "feature25",
  "feature27", "feature28", "feature29", "feature30", "feature31"
]

##B. selecting numeric variables
numerical_vars = ["feature1", "feature2","feature3", "feature4", "feature5",
                    "feature7", "feature8","feature9",
                   "feature10", "feature11","feature12", "feature13","feature14", "feature16",
                   "feature17", 'feature19',"feature20",
                  "feature18","feature21"
]

#%%
## All variables
all_vars = categorical_vars + numerical_vars


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Identify numerical and categorical variables
X_num = data[numerical_vars]  # Numerical features
X_cat = data[categorical_vars]  # Categorical features
y = data['outcome']  # Target variable

# Standardize numerical features
scaler = StandardScaler()
X_scaled_num = scaler.fit_transform(X_num)  # Scale only numerical variables

# One-Hot Encode categorical variables
encoder = OneHotEncoder(drop='first', sparse_output=False) 
X_encoded_cat = encoder.fit_transform(X_cat)

# Combine numerical and categorical features
X_transformed = np.hstack((X_scaled_num, X_encoded_cat))

# Get feature names after encoding
encoded_feature_names = encoder.get_feature_names_out(categorical_vars)
feature_names = list(numerical_vars) + list(encoded_feature_names)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42, stratify=y)

# Perform LASSO with cross-validation
lasso_cv = LassoCV(alphas=np.logspace(-6, 1, 100), cv=10, random_state=42)
lasso_cv.fit(X_train, y_train)

# Get the optimal lambda (alpha in sklearn)
optimal_lambda = lasso_cv.alpha_

# Extract coefficients at the optimal lambda
best_coefs = lasso_cv.coef_

# Select important features (non-zero coefficients)
selected_features = [feature for feature, coef in zip(feature_names, best_coefs) if coef != 0]

# Print Selected Features & Coefficients
print(f"Optimal Lambda (α): {optimal_lambda:.6f}")
print("\nSelected Features and Coefficients:")
for feature, coef in zip(selected_features, best_coefs[best_coefs != 0]):
    print(f"{feature}: {coef:.5f}")

# Plot LASSO CV Results (MSE vs. Lambda)
plt.figure(figsize=(8, 5))
plt.plot(np.log10(lasso_cv.alphas_), np.mean(lasso_cv.mse_path_, axis=1), marker='o', color='red')
plt.axvline(np.log10(optimal_lambda), linestyle="--", color='black', label=f"Optimal λ = {optimal_lambda:.5f}")
plt.xlabel("log(λ)")
plt.ylabel("Mean Squared Error")
plt.title("LASSO Cross-Validation Results")
plt.legend()
plt.show()

# Plot Feature Importance
importance_df = pd.DataFrame({"Feature": selected_features, "Coefficient": best_coefs[best_coefs != 0]})
importance_df = importance_df.sort_values(by="Coefficient", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Coefficient"], color='skyblue')
plt.xlabel("Coefficient Value")
plt.ylabel("Feature Name")
plt.title("Feature Importance - LASSO Regression")
plt.gca().invert_yaxis()
plt.show()


#%%

#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Standardize numerical variables
scaler = StandardScaler()
X = data[all_vars]
X_scaled = scaler.fit_transform(X[numerical_vars])
y = data['outcome']

# Set lambda (alpha) to 0.01
fixed_lambda = 0.0015

# Perform LASSO with the fixed lambda (alpha)
lasso = Lasso(alpha=fixed_lambda, random_state=42)
lasso.fit(X_scaled, y)

# Extract coefficients at the fixed lambda
fixed_lambda_coefs = lasso.coef_

# Get the list of selected features (non-zero coefficients)
selected_features_fixed_lambda = [
    feature for feature, coef in zip(X.columns, fixed_lambda_coefs) if coef != 0
]

# Print Selected Features & Coefficients
print(f"Fixed Lambda (α): {fixed_lambda}")
print("Selected Features and Coefficients:")
for feature, coef in zip(selected_features_fixed_lambda, fixed_lambda_coefs[fixed_lambda_coefs != 0]):
    print(f"{feature}: {coef: .5f}")

# Plot the coefficients path for different lambda values
alphas = np.logspace(-6, 1, 100)
coefs_path = []
for alpha in alphas:
    lasso_temp = Lasso(alpha=alpha, random_state=42)
    lasso_temp.fit(X_scaled, y)
    coefs_path.append(lasso_temp.coef_)

coefs_path = np.array(coefs_path).T

plt.figure(figsize=(8, 5))
for i in range(len(X.columns)):
    plt.plot(np.log10(alphas), coefs_path[i], label=X.columns[i])

plt.axvline(np.log10(fixed_lambda), linestyle="--", color='black', label=f"Fixed λ = {fixed_lambda:.5f}")
plt.xlabel("log(λ)")
plt.ylabel("Coefficients")
plt.title("LASSO Coefficient Paths")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%%
X_selected = ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11","feature13","feature14", "feature16", "feature17", "feature18","feature19", "feature20", "feature21", "feature24", "feature28"]

X_numeric = ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11","feature13","feature14", "feature17", "feature18","feature19", "feature20", "feature21"]

X_categorical = ["feature24", "feature28"]

X_category_one = ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11","feature14","feature24", "feature28"]

X_category_two = ["feature16", "feature17", "feature18","feature19", "feature20", "feature21"]

X_category_three = ["feature13", "feature16", "feature17","feature13", "feature18","feature19", "feature20"]

#%%
data = pd.read_csv(r"D:\Youtube channel\outcome_imputed.csv")

data.head()

data.drop(['Unnamed: 0', 'ID'], axis=1, inplace=True)

#%% Visualize outcome before sampling
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter(X, y, feature_x, feature_y):
    """
    Scatter plot of two selected features, colored by class labels (Green & Red).
    """
    plt.figure(figsize=(8, 6))

    # Custom color mapping: 0 = Green, 1 = Red
    palette = {0: "green", 1: "red"}

    sns.scatterplot(data=X, x=feature_x, y=feature_y, hue=y, palette=palette, alpha=0.7, edgecolor='k')

    plt.title(f"Scatter Plot of feature13 against feature14 by outcome (No sampling))")
    plt.xlabel("feature14")
    plt.ylabel("feature13")
    plt.legend(title=y.name)
    plt.grid(True)
    plt.show()

# Select features
feature_x = "feature13"
feature_y = "feature14"

# Extract the features and outcome
X = data[[feature_x, feature_y]]  # Only select the two numerical features
y = data['outcome'].astype(int)  # Ensure y is integer

# Plot scatter with green and red colors
plot_scatter(X, y, feature_x, feature_y)

#%% Visualize outcome distribution after SMOTE sampling
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
from numpy import where


X = data[all_vars]
y = data['outcome']

# summarize class distribution
counter = Counter(y)
print(counter)

# transform the data
oversample = SMOTE()
X,y = oversample.fit_resample(X,y)

# Count occurrences of each class in y
counter = Counter(y)
print(counter)

# Define color mapping: Class 0 -> Green, Class 1 -> Red
color_map = {0: "green", 1: "red"}


# Specify the exact features you want to plot
feature_x = "feature13"  # X-axis variable
feature_y = "feature14"  # Y-axis variable

# Scatter plot for each class
plt.figure(figsize=(8, 6))

for label, _ in counter.items():
    row_ix = np.where(y == label)[0]  # Get row indices for class
    plt.scatter(X.loc[row_ix, feature_x], X.loc[row_ix, feature_y],
                label=f"Class {label}",
                color=color_map[label],  # Assign class-specific colors
                alpha=0.7, edgecolor='black')

plt.legend()
plt.title(f"Scatter Plot of feature13 against. feature14 by outcome after SMOTE sampling")
plt.xlabel("feature14")
plt.ylabel("feature13")
plt.grid(True)
plt.show()

#%% Lets try adaptive sampling 

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import ADASYN  
import matplotlib.pyplot as plt
import numpy as np
from numpy import where

# Define features and target variable
X = data.drop(['outcome', 'feature32'], axis=1)
y = data['outcome']

# Summarize class distribution before resampling
counter_before = Counter(y)
print("Class distribution before resampling:", counter_before)

# Apply ADASYN for adaptive sampling
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# Summarize class distribution after resampling
counter_after = Counter(y_resampled)
print("Class distribution after resampling:", counter_after)

# Define color mapping: Class 0 -> Green, Class 1 -> Red
color_map = {0: "green", 1: "red"}

# Specify the exact features you want to plot
feature_x = "feature13"  # X-axis variable
feature_y = "feature14"    # Y-axis variable

# Scatter plot for each class
plt.figure(figsize=(8, 6))
for label, _ in counter_after.items():
    row_ix = np.where(y_resampled == label)[0]  # Get row indices for class
    plt.scatter(
        X_resampled.loc[row_ix, feature_x], 
        X_resampled.loc[row_ix, feature_y],
        label=f"Class {label}",
        color=color_map[label],  # Assign class-specific colors
        alpha=0.7, edgecolor='black'
    )

plt.legend()
plt.title(f"Scatter Plot of {feature_x} vs. {feature_y} by outcome (After ADASYN)")
plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.grid(True)
plt.show()

#%% visualizing outcome after smoteenn sampling
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.combine import SMOTEENN  # Import SMOTEENN for combined resampling
import matplotlib.pyplot as plt
import numpy as np
from numpy import where

# Define features and target variable
X = data.drop(['outcome', 'feature32'], axis=1)
y = data['outcome']

# Summarize class distribution before resampling
counter_before = Counter(y)
print("Class distribution before resampling:", counter_before)

# Apply SMOTEENN for combined resampling
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

# Summarize class distribution after resampling
counter_after = Counter(y_resampled)
print("Class distribution after resampling:", counter_after)

# Define color mapping: Class 0 -> Green, Class 1 -> Red
color_map = {0: "green", 1: "red"}

# Specify the exact features you want to plot
feature_x = "feature13"  # X-axis variable
feature_y = "feature14"    # Y-axis variable

# Scatter plot for each class
plt.figure(figsize=(8, 6))
for label, _ in counter_after.items():
    row_ix = np.where(y_resampled == label)[0]  # Get row indices for class
    plt.scatter(
        X_resampled.loc[row_ix, feature_x], 
        X_resampled.loc[row_ix, feature_y],
        label=f"Class {label}",
        color=color_map[label],  # Assign class-specific colors
        alpha=0.7, edgecolor='black'
    )

plt.legend()
plt.title(f"Scatter Plot of {feature_x} vs. {feature_y} by outcome (After SMOTEENN)")
plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.grid(True)
plt.show()


#%% Compare Models with no sampling method

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Define features and target variable
X = data[X_selected]  # Use selected features

y = data['outcome'].astype(int).to_numpy()

# Identify numerical and categorical features
X_numeric = data[X_numeric].columns
X_categorical = data[X_categorical].columns

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])  #scale only numericals

# Convert categorical features to NumPy array
X_categorical_np = X[X_categorical].to_numpy()

# Combine numerical and categorical features
X_transformed = np.hstack((X_scaled, X_categorical_np))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

# Define classifiers
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM (RBF Kernel)": SVC(probability=True, random_state=42),  # Enable probability for AUC
    "Naïve Bayes": GaussianNB()
}

# Store results
results = []

# Train & evaluate each model
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)

    # Predict probabilities (for AUC)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    # Compute AUC
    auc_score = roc_auc_score(y_test, y_pred_prob)

    # Compute Precision & Recall
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Store results
    results.append({"Model": name, "AUC": auc_score, "Precision": precision, "Recall": recall})

    # Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='cividis')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Convert results to DataFrame and display
results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
print("\n Model Performance Comparison:\n")
print(results_df)

# Plot AUC comparison
plt.figure(figsize=(8, 5))
plt.barh(results_df["Model"], results_df["AUC"], color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel("AUC Score")
plt.title("AUC Score Comparison Across Models")
plt.xlim(0, 1)
plt.grid(True)
plt.show()


#%% Apply SMOTE
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

# Define features and target variable
X = data[X_selected]  
y = data['outcome'].astype(int).to_numpy()

# Identify numerical and categorical features
X_numeric = data[X_numeric].columns
X_categorical = data[X_categorical].columns

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])  #scale only numericals

# Convert categorical features to NumPy array
X_categorical_np = X[X_categorical].to_numpy()

# Combine numerical and categorical features
X_transformed = np.hstack((X_scaled, X_categorical_np))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

# Apply SMOTE to handle class imbalance
# transform the data
oversample = SMOTE()

X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Define classifiers
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM (RBF Kernel)": SVC(probability=True, random_state=42),  # Enable probability for AUC
    "Naïve Bayes": GaussianNB()
}

# Store results
results = []

# Train & evaluate each model
for name, model in models.items():
    # Train model using SMOTE resampled dataset
    model.fit(X_train_resampled, y_train_resampled)

    # Predict probabilities for AUC (only for models that support probability estimates)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    # Compute AUC Score
    auc_score = roc_auc_score(y_test, y_pred_prob)

    # Compute Precision & Recall
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Store results
    results.append({"Model": name, "AUC": auc_score, "Precision": precision, "Recall": recall})

    # Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='cividis')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Convert results to DataFrame and display
results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
print("\n Model Performance Comparison:\n")
print(results_df)

# Plot AUC comparison
plt.figure(figsize=(8, 5))
plt.barh(results_df["Model"], results_df["AUC"], color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel("AUC Score")
plt.title("AUC Score Comparison Across Models (Using SMOTEENN)")
plt.xlim(0, 1)
plt.grid(True)
plt.show()




#%% Apply SMOTEENN
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.combine import SMOTEENN

# Define features and target variable
X = data[X_selected]  # Use selected features
y = data['outcome']
#.astype(int).to_numpy()

# Identify numerical and categorical features
X_numeric = data[X_numeric].columns
X_categorical = data[X_categorical].columns

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])  #scale only numericals

# Convert categorical features to NumPy array
X_categorical_np = X[X_categorical].to_numpy()

# Combine numerical and categorical features
X_transformed = np.hstack((X_scaled, X_categorical_np))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

# Apply SMOTEENN to handle class imbalance
smoteen = SMOTEENN()
X_train_resampled, y_train_resampled = smoteen.fit_resample(X_train, y_train)

# Define classifiers
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM (RBF Kernel)": SVC(probability=True, random_state=42),  # Enable probability for AUC
    "Naïve Bayes": GaussianNB()
}

# Store results
results = []

# Train & evaluate each model
for name, model in models.items():
    # Train model using SMOTEENN resampled dataset
    model.fit(X_train_resampled, y_train_resampled)

    # Predict probabilities for AUC (only for models that support probability estimates)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    # Compute AUC Score
    auc_score = roc_auc_score(y_test, y_pred_prob)

    # Compute Precision & Recall
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Store results
    results.append({"Model": name, "AUC": auc_score, "Precision": precision, "Recall": recall})

    # Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='cividis')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Convert results to DataFrame and display
results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
print("\n Model Performance Comparison:\n")
print(results_df)

# Plot AUC comparison
plt.figure(figsize=(8, 5))
plt.barh(results_df["Model"], results_df["AUC"], color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel("AUC Score")
plt.title("AUC Score Comparison Across Models (Using SMOTEENN)")
plt.xlim(0, 1)
plt.grid(True)
plt.show()


#%%
# Drop categorical variables

#%% Apply SMOTEENN
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.combine import SMOTEENN

# Define features and target variable
X = data[X_selected]  # Use selected features
y = data['outcome'].astype(int).to_numpy()


# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])  #scale only numericals


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTEENN to handle class imbalance
smoteen = SMOTEENN()
X_train_resampled, y_train_resampled = smoteen.fit_resample(X_train, y_train)

# Define classifiers
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM (RBF Kernel)": SVC(probability=True, random_state=42),  # Enable probability for AUC
    "Naïve Bayes": GaussianNB()
}

# Store results
results = []

# Train & evaluate each model
for name, model in models.items():
    # Train model using SMOTEENN resampled dataset
    model.fit(X_train_resampled, y_train_resampled)

    # Predict probabilities for AUC (only for models that support probability estimates)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    # Compute AUC Score
    auc_score = roc_auc_score(y_test, y_pred_prob)

    # Compute Precision & Recall
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Store results
    results.append({"Model": name, "AUC": auc_score, "Precision": precision, "Recall": recall})

    # Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='cividis')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Convert results to DataFrame and display
results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
print("\n Model Performance Comparison:\n")
print(results_df)

# Plot AUC comparison
plt.figure(figsize=(8, 5))
plt.barh(results_df["Model"], results_df["AUC"], color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel("AUC Score")
plt.title("AUC Score Comparison Across Models (Using SMOTEENN)")
plt.xlim(0, 1)
plt.grid(True)
plt.show()



#%% SMOTEEN AND ENSEMBLE
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.combine import SMOTEENN

# Define features and target variable
X = data[X_selected]  # Use selected features
y = data['outcome'].astype(int).to_numpy()

# Identify numerical and categorical features
X_numeric = data[X_numeric].columns
X_categorical = data[X_categorical].columns

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])  #scale only numericals

# Convert categorical features to NumPy array
X_categorical_np = X[X_categorical].to_numpy()

# Combine numerical and categorical features
X_transformed = np.hstack((X_scaled, X_categorical_np))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

# Apply SMOTEENN to handle class imbalance
smoteen = SMOTEENN()
X_train_resampled, y_train_resampled = smoteen.fit_resample(X_train, y_train)

# Define classifiers with and without Bagging
models = {
    "Bagging (Logistic Regression)": BaggingClassifier(estimator=LogisticRegression(), n_estimators=10, random_state=0),
    "Bagging (Decision Tree)": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=0),
    "Bagging (SVM)": BaggingClassifier(estimator=SVC(probability=True), n_estimators=10, random_state=0),
    "Bagging (Naïve Bayes)": BaggingClassifier(estimator=GaussianNB(), n_estimators=10, random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),  # RF already uses bagging implicitly
}

# Store results
results = []

# Train & evaluate each model
for name, model in models.items():
    # Train model using SMOTEENN resampled dataset
    model.fit(X_train_resampled, y_train_resampled)

    # Predict probabilities for AUC (only for models that support probability estimates)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    # Compute AUC Score
    auc_score = roc_auc_score(y_test, y_pred_prob)

    # Compute Precision & Recall
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Compute Precision-Recall AUC
    precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recalls, precisions)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")

    # Store results
    results.append({"Model": name, "AUC": auc_score, "Precision": precision, "Recall": recall, "P-R AUC": pr_auc})

    # Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='cividis')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Convert results to DataFrame and display
results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
print("\n Model Performance Comparison:\n")
print(results_df)

# Plot AUC comparison
plt.figure(figsize=(8, 5))
plt.barh(results_df["Model"], results_df["AUC"], color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel("AUC Score")
plt.title("AUC Score Comparison Across Bagging Models (Using SMOTEENN)")
plt.xlim(0, 1)
plt.grid(True)
plt.show()

#%%
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.combine import SMOTEENN

# Define features and target variable
X = data[X_selected]  # Use selected features
y = data['outcome'].astype(int).to_numpy()

# Identify numerical and categorical features
X_numeric = data[X_numeric].columns
X_categorical = data[X_categorical].columns

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])  #scale only numericals

# Convert categorical features to NumPy array
X_categorical_np = X[X_categorical].to_numpy()

# Combine numerical and categorical features
X_transformed = np.hstack((X_scaled, X_categorical_np))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

# Apply SMOTEENN to handle class imbalance
smoteen = SMOTEENN()
X_train_resampled, y_train_resampled = smoteen.fit_resample(X_train, y_train)

# Define classifiers with and without Bagging
models = {
    "Bagging (Logistic Regression)": BaggingClassifier(estimator=LogisticRegression(), n_estimators=10, random_state=0),
    "Bagging (Decision Tree)": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=0),
    "Bagging (SVM)": BaggingClassifier(estimator=SVC(probability=True), n_estimators=10, random_state=0),
    "Bagging (Naïve Bayes)": BaggingClassifier(estimator=GaussianNB(), n_estimators=10, random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),  # RF already uses bagging implicitly
}

# Store results
results = []

# Train & evaluate each model
for name, model in models.items():
    # Train model using SMOTEENN resampled dataset
    model.fit(X_train_resampled, y_train_resampled)

    # Predict probabilities for AUC (only for models that support probability estimates)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    # Compute AUC Score
    auc_score = roc_auc_score(y_test, y_pred_prob)

    # Compute Precision & Recall
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Compute Precision-Recall AUC
    precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recalls, precisions)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")

    # Store results
    results.append({"Model": name, "AUC": auc_score, "Precision": precision, "Recall": recall, "P-R AUC": pr_auc})

    # Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='cividis')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Convert results to DataFrame and display
results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
print("\n Model Performance Comparison:\n")
print(results_df)

# Plot AUC comparison
plt.figure(figsize=(8, 5))
plt.barh(results_df["Model"], results_df["AUC"], color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel("AUC Score")
plt.title("AUC Score Comparison Across Bagging Models (Using SMOTEENN)")
plt.xlim(0, 1)
plt.grid(True)
plt.show()

#%%
## Lets concetrate on Logistic Regression, SVM,  Naive Bayes, Random forest, 

#%%

## STARTING WITH LOGISTIC REGRESSION

#%%  No sampling method, only class weight change

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score

## define x and y variables
X = data[X_selected]
y = data['outcome']

X_scaled = StandardScaler().fit_transform(X[X_numeric])

## split training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


## fit model
model_logistic = LogisticRegression(class_weight= "balanced")
model_logistic.fit(X_train, y_train)

## predict probabilities
y_pred_prob = model_logistic.predict_proba(X_test)[:, 1]

# Compute AUC Score
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"Logistic AUC Score: {auc_score:.4f}")

## Compute precision and recall
precision = precision_score(y_test, model_logistic.predict(X_test))
recall = recall_score(y_test, model_logistic.predict(X_test))

print(f"Logistic Precision: {precision:.4f}")
print(f"Logistic Recall: {recall:.4f}")

# Compute Precision-Recall AUC
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)
print(f"Precision-Recall AUC: {pr_auc:.4f}")


#%% Applying SMOTEENN sampling with Bagging
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, precision_recall_curve, auc
)
from imblearn.combine import SMOTEENN
from sklearn.ensemble import BaggingClassifier

# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

X_scaled = StandardScaler().fit_transform(X[X_numeric])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# Apply SMOTEENN to handle class imbalance
smoteen = SMOTEENN()
X_train_resampled, y_train_resampled = smoteen.fit_resample(X_train, y_train)

# Fit Bagging with Logistic Regression
model_logistic = BaggingClassifier(estimator=LogisticRegression(), n_estimators=10, random_state=0)
model_logistic.fit(X_train_resampled, y_train_resampled)

# Predict probabilities
y_pred_prob = model_logistic.predict_proba(X_test)[:, 1]

# Compute AUC Score
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"Logistic ROC-AUC Score: {auc_score:.4f}")

# Compute Precision and Recall
y_pred = model_logistic.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Logistic Precision: {precision:.4f}")
print(f"Logistic Recall: {recall:.4f}")

# Compute Precision-Recall AUC
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Logistic Regression (Bagging)")
plt.legend()
plt.grid(True)
plt.show()


#%% Applying smoteenn with Bagging and RepeatedstratifiedKFold

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, make_scorer
)
from imblearn.combine import SMOTEENN
from sklearn.ensemble import BaggingClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

# scale numercal variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])

# Define pipeline: Scaling, SMOTEENN (inside), and Model
# Use ImbPipeline instead of Pipeline to handle SMOTEENN correctly
pipeline = ImbPipeline([
    ('scaler', StandardScaler()), 
    ('resampling', SMOTEENN()),  
    ('model', BaggingClassifier(estimator=LogisticRegression(solver='liblinear'), n_estimators=10, random_state=0))  # Model
])

# Define cross-validation strategy: 10-fold CV with 3 repeats
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

# Define scoring metrics
scoring = {
    'roc_auc': 'roc_auc',
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score)
}

# Perform cross-validation
cv_results = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
mean_auc = np.mean(cv_results)

print(f"Logistic Bagging CV AUC Score: {mean_auc:.4f}")

# Compute precision and recall with cross-validation
cv_precision = cross_val_score(pipeline, X, y, cv=cv, scoring='precision', n_jobs=-1).mean()
cv_recall = cross_val_score(pipeline, X, y, cv=cv, scoring='recall', n_jobs=-1).mean()

print(f"Logistic Bagging CV Precision: {cv_precision:.4f}")
print(f"Logistic Bagging CV Recall: {cv_recall:.4f}")

# Fit pipeline on entire dataset for final Precision-Recall AUC
pipeline.fit(X, y)
y_pred_prob = pipeline.predict_proba(X)[:, 1]

# Compute Precision-Recall AUC
precisions, recalls, _ = precision_recall_curve(y, y_pred_prob)
pr_auc = auc(recalls, precisions)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Logistic Regression (SMOTEEN Bagging) - CV")
plt.legend()
plt.grid(True)
plt.show()

#%% Aplying SMOTE sampling

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, precision_recall_curve, auc
)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaggingClassifier

# Define features and target variable
X = data[X_selected]
y = data['outcome']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE to handle class imbalance
oversample = SMOTE()
X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Fit Bagging with Logistic Regression
model_logistic = BaggingClassifier(estimator=LogisticRegression(), n_estimators=10, random_state=0)
model_logistic.fit(X_train_resampled, y_train_resampled)

# Predict probabilities
y_pred_prob = model_logistic.predict_proba(X_test)[:, 1]

# Compute AUC Score
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"Logistic ROC-AUC Score: {auc_score:.4f}")

# Compute Precision and Recall
y_pred = model_logistic.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Logistic Precision: {precision:.4f}")
print(f"Logistic Recall: {recall:.4f}")

# Compute Precision-Recall AUC
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Logistic Regression (SMOTE Bagging)")
plt.legend()
plt.grid(True)
plt.show()


#%% Gridsearch
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, precision_recall_curve, auc, make_scorer
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # pipeline
from sklearn.ensemble import BaggingClassifier

# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

# Define pipeline: Scaling, SMOTE (inside), and Model
pipeline = ImbPipeline([
    ('scaler', StandardScaler()), 
    ('resampling', SMOTE()),  
    ('model', BaggingClassifier(estimator=LogisticRegression(solver='liblinear', max_iter=500),
                                n_estimators=10, random_state=0))  # Model
])

# Define hyperparameter grid for Logistic Regression inside BaggingClassifier
param_grid = {
   'model__estimator__C': [0.01, 0.1, 1, 10],
   'model__estimator__penalty': ['l1', 'l2']  # Must apply inside `estimator`
}

# Define cross-validation strategy: 10-fold CV with 3 repeats
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

# Grid search with Precision-Recall AUC scoring
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='average_precision', n_jobs=-1)

# Fit Grid Search on the dataset
grid_search.fit(X, y)

# Print best hyperparameters
print("Best Parameters:", grid_search.best_params_)

# Print best Precision-Recall AUC from Grid Search
print("Best Precision-Recall AUC:", grid_search.best_score_)

# Get best model and predict probabilities
y_pred_prob = grid_search.best_estimator_.predict_proba(X)[:, 1]

# Compute Precision-Recall AUC
precisions, recalls, _ = precision_recall_curve(y, y_pred_prob)
pr_auc = auc(recalls, precisions)
print(f"Final Precision-Recall AUC: {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Logistic Regression (Bagging) - CV")
plt.legend()
plt.grid(True)
plt.show()


#%% Implementing best parameters
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, auc, accuracy_score, roc_curve
)

# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])

# Train Logistic Regression Model with Best Parameters
best_model = LogisticRegression(
    C=0.01,
    class_weight={0:28, 1: 35},
    fit_intercept=False,
    penalty='l1',
    solver='liblinear',
    max_iter=500
)

# Split dataset 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Over sample using SMOTE
oversample = SMOTE()

X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Fit the model
best_model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities for ROC and Precision-Recall AUC
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Predict class labels
y_pred = best_model.predict(X_test)

# Compute Performance Metrics
roc_auc = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Compute Precision-Recall AUC
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)

# Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Print Results
print(f" **Model Performance** ")
print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap='cividis')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for reference
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


#%%

# FITTING DIFFERENT FEATURES

#%% Fitting different features With no sampling applied
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, accuracy_score,
    precision_recall_curve, auc, confusion_matrix
)

# Define multiple feature sets
feature_sets = {
"X_selected ": ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11","feature13","feature14", "feature16", "feature17", "feature18","feature19", "feature20", "feature21"],

"X_category_one ": ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11"],

"X_feature16" : ["feature16"],

"X_category_two ": ["feature16", "feature18", "feature17", "feature18","feature19"],

"X_category_three ": ["feature13", "feature13", "feature20", "feature14", "feature9"]
    

    
}

# Define target variable
y = data['outcome'].astype(int).to_numpy()

# Store results
results = []

# Iterate over different feature sets
for name, features in feature_sets.items():
    # Select X variables
    X = data[features]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Logistic Regression Model
    model = LogisticRegression(
       C=0.01,
       class_weight={0:28, 1: 35},
       fit_intercept=False,
       penalty='l1',
       solver='liblinear',
       max_iter=500
    )
    
    # Fit model
    model.fit(X_scaled, y)
    
    # Predict probabilities for ROC and Precision-Recall AUC
    y_pred_prob = model.predict_proba(X_scaled)[:, 1]
    
    # Predict class labels
    y_pred = model.predict(X_scaled)
    
    # Compute Metrics
    roc_auc = roc_auc_score(y, y_pred_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    
    # Compute Precision-Recall AUC
    precisions, recalls, _ = precision_recall_curve(y, y_pred_prob)
    pr_auc = auc(recalls, precisions)
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    # Compute False Positive Rate (FPR)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Store results
    results.append([name, roc_auc, pr_auc, accuracy, precision, recall, fpr])
    

# Convert results to DataFrame
metrics_df = pd.DataFrame(results, columns=["Feature Set", "ROC AUC", "PR AUC", "Accuracy", "Precision", "Recall", "FPR"])

# Display the table
# Option 1: Use print() for simple output
print("Logistic Regression Performance Metrics:")
print(metrics_df)

# Option 2: Use display() in Jupyter Notebook for better formatting
try:
    from IPython.display import display
    print("\nLogistic Regression Performance Metrics (Formatted):")
    display(metrics_df)
except ImportError:
    pass

# Option 3: Export to HTML for sharing or viewing in a browser
metrics_df.to_html("logistic_regression_metrics.html")
print("\nPerformance metrics have been saved to 'logistic_regression_metrics.html'. Open this file in a browser for a clean view.")


#%% Fitting different features With SMOTE sampling applied

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, auc, accuracy_score)

# Define multiple feature sets
feature_sets = {
"X_selected ": ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11","feature13","feature14", "feature16", "feature17", "feature18","feature19", "feature20", "feature21"],

"X_category_one ": ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11"],

"X_feature16" : ["feature16"],

"X_category_two ": ["feature16", "feature18", "feature17", "feature18","feature19"],

"X_category_three ": ["feature13", "feature20", "feature14", "feature9"]
    

    
}

# Define target variable
y = data['outcome'].astype(int).to_numpy()

# Store results
results = []

# Iterate over different feature sets
for name, features in feature_sets.items():
    # Select X variables
    X = data[features].values  
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Apply SMOTE to handle class imbalance
    oversample = SMOTE(sampling_strategy='minority', k_neighbors=10)
    X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)
    
    
    # Train Logistic Regression Model
    model = LogisticRegression(
        C=0.01, 
        class_weight={0:28, 1: 35}, 
        fit_intercept=False, 
        penalty='l1', 
        solver='liblinear', 
        max_iter=500
    )
    
    # Fit model
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict probabilities for ROC and Precision-Recall AUC
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Predict class labels
    y_pred = model.predict(X_test)
    
    # Compute Metrics
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Compute Precision-Recall AUC
    precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recalls, precisions)
    
   # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Compute False Positive Rate (FPR)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Store results
    results.append([name, roc_auc, pr_auc, accuracy, precision, recall, fpr])
    

# Convert results to DataFrame
metrics_df = pd.DataFrame(results, columns=["Feature Set", "ROC AUC", "PR AUC", "Accuracy", "Precision", "Recall", "FPR"])

# Display the table
# Option 1: Use print() for simple output
print("Logistic Regression Performance Metrics:")
print(metrics_df)

# Option 2: Use display() in Jupyter Notebook for better formatting
try:
    from IPython.display import display
    print("\nLogistic Regression Performance Metrics (Formatted):")
    display(metrics_df)
except ImportError:
    pass

# Option 3: Export to HTML for sharing or viewing in a browser
metrics_df.to_html("logistic_regression_metrics.html")
print("\nPerformance metrics have been saved to 'logistic_regression_metrics.html'. Open this file in a browser for a clean view.")

#%%

#NAIVE BAYES CLASSIFIER

#%% Naive Bayes with SMOTE
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB  
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, precision_recall_curve, auc
)
from imblearn.over_sampling import SMOTE


# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE to handle class imbalance
oversample = SMOTE()
X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# FitNaive Bayes
model_nb = GaussianNB()  # Use GaussianNB
model_nb.fit(X_train_resampled, y_train_resampled)

# Predict probabilities
y_pred_prob = model_nb.predict_proba(X_test)[:, 1]

# Compute AUC Score
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"Naive Bayes ROC-AUC Score: {auc_score:.4f}")

# Compute Precision and Recall
y_pred = model_nb.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Naive Bayes Precision: {precision:.4f}")
print(f"Naive Bayes Recall: {recall:.4f}")

# Compute Precision-Recall AUC
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Naive Bayes (SMOTE")
plt.legend()
plt.grid(True)
plt.show()


#%% Naive Bayes with SMOTE
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB  
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, precision_recall_curve, auc
)
from imblearn.over_sampling import SMOTE


# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance
oversample = SMOTE(sampling_strategy='auto', k_neighbors=10, random_state=42)
X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# FitNaive Bayes
model_nb = GaussianNB()  # Use GaussianNB
model_nb.fit(X_train_resampled, y_train_resampled)

# Predict probabilities
y_pred_prob = model_nb.predict_proba(X_test)[:, 1]

# Compute AUC Score
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"Naive Bayes ROC-AUC Score: {auc_score:.4f}")

# Compute Precision and Recall
y_pred = model_nb.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Naive Bayes Precision: {precision:.4f}")
print(f"Naive Bayes Recall: {recall:.4f}")

# Compute Precision-Recall AUC
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Naive Bayes (SMOTE)")
plt.legend()
plt.grid(True)
plt.show()



#%% Naive Bayes with Bagging and SMOTE
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, precision_recall_curve, auc
)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaggingClassifier

# Define features and target variable
X = data[X_selected]
y = data['outcome']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE to handle class imbalance
oversample = SMOTE()
X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Fit Bagging with Naive Bayes
model_nb = BaggingClassifier(estimator=GaussianNB(), n_estimators=10, random_state=0)  # Use GaussianNB
model_nb.fit(X_train_resampled, y_train_resampled)

# Predict probabilities
y_pred_prob = model_nb.predict_proba(X_test)[:, 1]

# Compute AUC Score
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"Naive Bayes ROC-AUC Score: {auc_score:.4f}")

# Compute Precision and Recall
y_pred = model_nb.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Naive Bayes Precision: {precision:.4f}")
print(f"Naive Bayes Recall: {recall:.4f}")

# Compute Precision-Recall AUC
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Naive Bayes (SMOTE Bagging)")
plt.legend()
plt.grid(True)
plt.show()

#%% NB parameter grid search
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaggingClassifier

# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_selected])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE to handle class imbalance
oversample = SMOTE()
X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [5, 10, 20, 50],
    'max_samples': [0.5, 0.7, 1.0],
    'bootstrap': [True, False]
}

# Initialize the Bagging classifier with GaussianNB as base estimator
bagging_nb = BaggingClassifier(estimator=GaussianNB(), random_state=0)

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(bagging_nb, param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Display best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)


#%% Implementing best parameters
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaggingClassifier

# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_selected])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance
oversample = SMOTE(sampling_strategy='auto', k_neighbors=10, random_state=42)
X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Define and train the Bagging classifier with the best parameters
best_model = BaggingClassifier(
    estimator=GaussianNB(priors=[0.1, 0.9]), 
    n_estimators=5, 
    max_samples=0.7, 
    bootstrap=True, 
    random_state=1
)
best_model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities and labels
y_pred_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# Compute evaluation metrics
auc_score = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)

# Print metrics
print(f"Naive Bayes Bagging ROC-AUC Score: {auc_score:.4f}")
print(f"Naive Bayes Bagging Precision: {precision:.4f}")
print(f"Naive Bayes Bagging Recall: {recall:.4f}")
print(f"Naive Bayes Bagging F1 Score: {f1:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Naive Bayes (SMOTE Bagging)")
plt.legend()
plt.grid(True)
plt.show()

# Compute and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

#%%

# CATEGORICAL NAIVE BAYES 

#%% 


#%%

#SVM

#%%

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, auc, accuracy_score, roc_curve
)

# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])

# Train Logistic Regression Model with Best Parameters
best_model = SVC(probability=True, random_state=42)

# Split dataset 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Over sample using SMOTE
oversample = SMOTE()

X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Fit the model
best_model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities for ROC and Precision-Recall AUC
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Predict class labels
y_pred = best_model.predict(X_test)

# Compute Performance Metrics
roc_auc = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Compute Precision-Recall AUC
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)

# Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Print Results
print(f" **Model Performance** ")
print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap='cividis')
plt.title("Confusion Matrix - SVM")
plt.show()

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for SVM")
plt.legend()
plt.grid(True)
plt.show()

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for reference
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


#%%
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SVMSMOTE 
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, auc, accuracy_score, roc_curve
)

# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])

# Train Logistic Regression Model with Best Parameters
best_model = SVC(probability=True, random_state=42)

# Split dataset 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Over sample using SMOTE
oversample = SVMSMOTE()

X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Fit the model
best_model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities for ROC and Precision-Recall AUC
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Predict class labels
y_pred = best_model.predict(X_test)

# Compute Performance Metrics
roc_auc = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Compute Precision-Recall AUC
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)

# Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Print Results
print(f" **Model Performance** ")
print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap='cividis')
plt.title("Confusion Matrix - SVM")
plt.show()

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for SVM")
plt.legend()
plt.grid(True)
plt.show()

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'SVM ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for reference
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


#%%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    precision_recall_curve,
    auc,
)
from imblearn.over_sampling import SVMSMOTE  
from sklearn.svm import SVC

# Define multiple feature sets
feature_sets = {
"X_selected ": ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11","feature13","feature14", "feature16", "feature17", "feature18","feature19", "feature20", "feature21"],

"X_category_one ": ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11"],

"X_feature16" : ["feature16"],

"X_category_two ": ["feature16", "feature18", "feature17", "feature18","feature19"],

"X_category_three ": ["feature13", "feature20", "feature14", "feature9"]
    

    
}

# Define target variable
y = data['outcome'].values  

# Store results
results = []

# Iterate over different feature sets
for name, features in feature_sets.items():
    # Select X variables
    X = data[features].values  
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Apply SVMSMOTE to handle class imbalance
    oversample = SVMSMOTE()
    X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)
    
    # train svm model
    model = SVC(probability=True, random_state=42)
        
    # Fit model
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict probabilities for ROC and Precision-Recall AUC
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Set adjustable threshold
    threshold = 0.5  # Change this value to adjust the decision boundary
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Compute Metrics
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Compute Precision-Recall AUC
    precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recalls, precisions)
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Compute False Positive Rate (FPR)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Store results
    results.append([name, roc_auc, pr_auc, accuracy, precision, recall, fpr])
    

# Convert results to DataFrame
metrics_df = pd.DataFrame(results, columns=["Feature Set", "ROC AUC", "PR AUC", "Accuracy", "Precision", "Recall", "FPR"])

# Display the table
print("SVM Performance Metrics:")
print(metrics_df)

# Export to HTML for sharing or viewing in a browser
metrics_df.to_html("SVM_forest_metrics.html")
print("\nPerformance metrics have been saved to 'SVM_metrics.html'. Open this file in a browser for a clean view.")

#%%

#%%

# FITTING A RANDOM FOREST MODEL

#%% Random forest with smote
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE to handle class imbalance
oversample = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Define and train the Random Forest classifier
best_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None, 
    class_weight='balanced', 
    random_state=0
)
best_model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities and labels
y_pred_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# Compute evaluation metrics
auc_score = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)

# Print metrics
print(f"Random Forest ROC-AUC Score: {auc_score:.4f}")
print(f"Random Forest Precision: {precision:.4f}")
print(f"Random Forest Recall: {recall:.4f}")
print(f"Random Forest F1 Score: {f1:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Random Forest (SMOTE)")
plt.legend()
plt.grid(True)
plt.show()

# Compute and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

#%% RF Parameter grid search

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE to handle class imbalance
oversample = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=0)

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(rf, param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get best parameters and train final model
best_params = grid_search.best_params_
best_model = RandomForestClassifier(**best_params, random_state=0)
best_model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities and labels
y_pred_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# Compute evaluation metrics
auc_score = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)

# Print metrics
print("Best Parameters:", best_params)
print(f"Random Forest ROC-AUC Score: {auc_score:.4f}")
print(f"Random Forest Precision: {precision:.4f}")
print(f"Random Forest Recall: {recall:.4f}")
print(f"Random Forest F1 Score: {f1:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Random Forest (SMOTE)")
plt.legend()
plt.grid(True)
plt.show()

# Compute and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

#%% Implementing best parameters

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from imblearn.over_sampling import SMOTE

# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE to handle class imbalance
oversample = SMOTE(sampling_strategy='auto', k_neighbors=10, random_state=42)
X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Define and train the Random Forest classifier
best_model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=20, 
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight={0: 10, 1: 5}, 
    random_state=0
)
best_model.fit(X_train_resampled, y_train_resampled)


# Predict probabilities and labels
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Set adjustable threshold
threshold = 0.3  # Change this value to adjust the decision boundary
y_pred = (y_pred_prob >= threshold).astype(int)

# Compute evaluation metrics
auc_score = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)

# Print metrics
print(f"Random Forest ROC-AUC Score: {auc_score:.4f}")
print(f"Random Forest Precision: {precision:.4f}")
print(f"Random Forest Recall: {recall:.4f}")
print(f"Random Forest F1 Score: {f1:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Random Forest (SMOTE)")
plt.legend()
plt.grid(True)
plt.show()

# Compute and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix-RF")
plt.show()

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'RF ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for reference
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RF Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
)

# Define multiple feature sets
feature_sets = {
"X_selected ": ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11","feature13","feature14", "feature16", "feature17", "feature18","feature19", "feature20", "feature21"],

"X_category_one ": ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11"],

"X_feature16" : ["feature16"],

"X_category_two ": ["feature16", "feature18", "feature17", "feature18","feature19"],

"X_category_three ": ["feature13", "feature20", "feature14", "feature9"]
    

    
}

# Define target variable
y = data['outcome'].astype(int).to_numpy()

# Store results
results = []

# Iterate over different feature sets
for name, features in feature_sets.items():
    # Select X variables
    X = data[features]  
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Apply SMOTE to handle class imbalance
    oversample = SMOTE(sampling_strategy='auto', k_neighbors=10, random_state=42)
    X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)
    
    # Train Random Forest Model
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=20, 
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight={0: 10, 1: 5}, 
        random_state=0
    )
    
    # Fit model
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict probabilities for ROC and Precision-Recall AUC
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Set adjustable threshold
    threshold = 0.5  # Change this value to adjust the decision boundary
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Compute Metrics
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Compute Precision-Recall AUC
    precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recalls, precisions)
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Compute False Positive Rate (FPR)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Store results
    results.append([name, roc_auc, pr_auc, accuracy, precision, recall, fpr])
    

# Convert results to DataFrame
metrics_df = pd.DataFrame(results, columns=["Feature Set", "ROC AUC", "PR AUC", "Accuracy", "Precision", "Recall", "FPR"])

# Display the table
print("Random Forest Performance Metrics:")
print(metrics_df)

# Export to HTML for sharing or viewing in a browser
metrics_df.to_html("random_forest_metrics.html")
print("\nPerformance metrics have been saved to 'random_forest_metrics.html'. Open this file in a browser for a clean view.")

#%%

## GRADIENT BOOSTING

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
)

# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE to handle class imbalance
oversample = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Gradient Boosting classifier
gbm = GradientBoostingClassifier(random_state=0)

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(gbm, param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get best parameters and train final model
best_params = grid_search.best_params_
best_model = GradientBoostingClassifier(**best_params, random_state=0)
best_model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities and labels
y_pred_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# Compute evaluation metrics
auc_score = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)

# Print metrics
print("Best Parameters:", best_params)
print(f"Gradient Boosting ROC-AUC Score: {auc_score:.4f}")
print(f"Gradient Boosting Precision: {precision:.4f}")
print(f"Gradient Boosting Recall: {recall:.4f}")
print(f"Gradient Boosting F1 Score: {f1:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Gradient Boosting (SMOTE)")
plt.legend()
plt.grid(True)
plt.show()

# Compute and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_curve
)

# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE to handle class imbalance
oversample = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Define and train the Gradient Boosting classifier
best_model = GradientBoostingClassifier(
    n_estimators=1000,  # Increased estimators to learn better
    learning_rate=0.2,  # Increased learning rate for faster correction
    max_depth=12,  # Deeper trees to capture more patterns
    min_samples_split=8,
    min_samples_leaf=2,
    subsample=0.8,
    loss='exponential',  # Using exponential loss to penalize wrong class 1 predictions more
    random_state=0
)

best_model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities and labels
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Set adjustable threshold
threshold = np.percentile(y_pred_prob, 75)
y_pred = (y_pred_prob >= threshold).astype(int)


# Compute evaluation metrics
auc_score = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)

# Print metrics
print(f"Gradient Boosting ROC-AUC Score: {auc_score:.4f}")
print(f"Gradient Boosting Precision: {precision:.4f}")
print(f"Gradient Boosting Recall: {recall:.4f}")
print(f"Gradient Boosting F1 Score: {f1:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Gradient Boosting (SMOTE)")
plt.legend()
plt.grid(True)
plt.show()

# Compute and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Plot histogram of predicted probabilities for class 1
plt.figure(figsize=(8, 6))
plt.hist(y_pred_prob, bins=30, color='blue', alpha=0.7)
plt.xlabel("Predicted Probability of Class 1")
plt.ylabel("Frequency")
plt.title("Histogram of Predicted Probabilities for Class 1")
plt.grid(True)
plt.show()

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'GBM ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for reference
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('GBM Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
)

# Define features and target variable
X = data[X_selected]
y = data['outcome']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE to handle class imbalance
oversample = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Define and train the Gradient Boosting classifier with increased class weight penalty
best_model = GradientBoostingClassifier(
    n_estimators=1000,  # Increased estimators to learn better
    learning_rate=0.2,  # Increased learning rate for faster correction
    max_depth=12,  # Deeper trees to capture more patterns
    min_samples_split=8,
    min_samples_leaf=2,
    subsample=0.8,
    loss='exponential',  # Using exponential loss to penalize wrong class 1 predictions more
    random_state=0
)
best_model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities and labels
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Adjust threshold dynamically based on quantiles to improve class 1 recall
threshold = np.percentile(y_pred_prob, 75)  # Setting threshold at the 75th percentile
y_pred = (y_pred_prob >= threshold).astype(int)

# Compute evaluation metrics
auc_score = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)

# Print metrics
print(f"Gradient Boosting ROC-AUC Score: {auc_score:.4f}")
print(f"Gradient Boosting Precision: {precision:.4f}")
print(f"Gradient Boosting Recall: {recall:.4f}")
print(f"Gradient Boosting F1 Score: {f1:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Gradient Boosting (SMOTE)")
plt.legend()
plt.grid(True)
plt.show()

# Compute and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Plot histogram of predicted probabilities for class 1
plt.figure(figsize=(8, 6))
plt.hist(y_pred_prob, bins=30, color='blue', alpha=0.7)
plt.xlabel("Predicted Probability of Class 1")
plt.ylabel("Frequency")
plt.title("Histogram of Predicted Probabilities for Class 1")
plt.grid(True)
plt.show()

# Display feature importance to identify key predictive features
feature_importance = best_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(X_selected)), feature_importance, tick_label=X_selected, color='blue', alpha=0.7)
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance from Gradient Boosting Model")
plt.grid(True)
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
)

# Define multiple feature sets
feature_sets = {
"X_selected ": ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11","feature13","feature14", "feature16", "feature17", "feature18","feature19", "feature20", "feature21"],

"X_category_one ": ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11"],

"X_feature16" : ["feature16"],

"X_category_two ": ["feature16", "feature18", "feature17", "feature18","feature19"],

"X_category_three ": ["feature13", "feature20", "feature14", "feature9"]
    

    
}

# Define target variable
y = data['outcome'].astype(int).to_numpy()  

# Store results
results = []

# Iterate over different feature sets
for name, features in feature_sets.items():
    # Select X variables
    X = data[features]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Apply SMOTE to handle class imbalance
    oversample = SMOTE(sampling_strategy="minority", k_neighbors=5, random_state=42)
    X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)
    
    # Train Gradient Boosting Model
    model = GradientBoostingClassifier(
        n_estimators=1000,  
        learning_rate=0.2,  
        max_depth=12, 
        min_samples_split=8,
        min_samples_leaf=2,
        subsample=0.8,
        loss='exponential',  # Using exponential loss to penalize wrong class 1 predictions more
        random_state=0
    )
    
    # Fit model
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict probabilities for ROC and Precision-Recall AUC
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Set adjustable threshold
    threshold = np.percentile(y_pred_prob, 75) # Change this value to adjust the decision boundary
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Compute Metrics
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Compute Precision-Recall AUC
    precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recalls, precisions)
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Compute False Positive Rate (FPR)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Store results
    results.append([name, roc_auc, pr_auc, accuracy, precision, recall, fpr])
    

# Convert results to DataFrame
metrics_df = pd.DataFrame(results, columns=["Feature Set", "ROC AUC", "PR AUC", "Accuracy", "Precision", "Recall", "FPR"])

# Display the table
print("Gradient Boosting Performance Metrics:")
print(metrics_df)

# Export to HTML for sharing or viewing in a browser
metrics_df.to_html("gradient_boosting_metrics.html")
print("\nPerformance metrics have been saved to 'gradient_boosting_metrics.html'. Open this file in a browser for a clean view.")


#%%

# XGBoost

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
)

# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE to handle class imbalance
oversample = SMOTE()
X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [200, 300, 400],
    'learning_rate': [0.1, 0.3, 0.5],
    'max_depth': [10, 15, 20],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [5, 10, 15]  # Penalizing false negatives
}

# Initialize the XGBoost classifier
xgb = XGBClassifier(eval_metric='logloss', random_state=0)

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(xgb, param_grid, scoring='recall', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get best parameters and train final model
best_params = grid_search.best_params_
best_model = XGBClassifier(**best_params, eval_metric='logloss', random_state=0)
best_model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities and labels
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Adjust threshold dynamically to reduce false negatives
threshold = 0.5  # Lowering threshold to favor recall
y_pred = (y_pred_prob >= threshold).astype(int)

# Compute evaluation metrics
auc_score = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)

# Print metrics
print("Best Parameters:", best_params)
print(f"XGBoost ROC-AUC Score: {auc_score:.4f}")
print(f"XGBoost Precision: {precision:.4f}")
print(f"XGBoost Recall: {recall:.4f}")
print(f"XGBoost F1 Score: {f1:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for XGBoost (SMOTE)")
plt.legend()
plt.grid(True)
plt.show()

# Compute and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Plot histogram of predicted probabilities for class 1
plt.figure(figsize=(8, 6))
plt.hist(y_pred_prob, bins=30, color='blue', alpha=0.7)
plt.xlabel("Predicted Probability of Class 1")
plt.ylabel("Frequency")
plt.title("Histogram of Predicted Probabilities for Class 1")
plt.grid(True)
plt.show()

# Display feature importance to identify key predictive features
feature_importance = best_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(X_selected)), feature_importance, tick_label=X_selected, color='blue', alpha=0.7)
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance from XGBoost Model")
plt.grid(True)
plt.show()

#%%

## Implementing the best parameters 

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_curve
)

# Define features and target variable
X = data[X_selected]
y = data['outcome'].astype(int).to_numpy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[X_numeric])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE to handle class imbalance
oversample = SMOTE()
X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Initialize the XGBoost classifier
#xgb = XGBClassifier(eval_metric='logloss', random_state=0)

# # Perform Grid Search with 5-fold cross-validation
# grid_search = GridSearchCV(xgb, param_grid, scoring='recall', cv=5, n_jobs=-1, verbose=1)
# grid_search.fit(X_train_resampled, y_train_resampled)

# Get best parameters and train final model
#best_params = grid_search.best_params_
best_model = XGBClassifier(
   scale_pos_weight=15,
   colsample_bytree=0.8, 
   learning_rate=0.1, 
   max_depth=15, 
   min_child_weight=1, 
   n_estimators=200, 
   subsample=0.6, 
   eval_metric='logloss', 
   random_state=0
)
best_model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities and labels
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Adjust threshold dynamically to reduce false negatives
threshold = 0.3 # Lowering threshold to favor recall
y_pred = (y_pred_prob >= threshold).astype(int)

# Compute evaluation metrics
auc_score = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)

# Print metrics
#print("Best Parameters:", best_params)
print(f"XGBoost ROC-AUC Score: {auc_score:.4f}")
print(f"XGBoost Precision: {precision:.4f}")
print(f"XGBoost Recall: {recall:.4f}")
print(f"XGBoost F1 Score: {f1:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for XGBoost (SMOTE)")
plt.legend()
plt.grid(True)
plt.show()

# Compute and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("XGBoost Confusion Matrix")
plt.show()

# Plot histogram of predicted probabilities for class 1
plt.figure(figsize=(8, 6))
plt.hist(y_pred_prob, bins=30, color='blue', alpha=0.7)
plt.xlabel("Predicted Probability of Class 1")
plt.ylabel("Frequency")
plt.title("Histogram of Predicted Probabilities for Class 1")
plt.grid(True)
plt.show()

# Display feature importance to identify key predictive features
# feature_importance = best_model.feature_importances_
# plt.figure(figsize=(10, 6))
# plt.bar(range(len(X_selected)), feature_importance, tick_label=X_selected, color='blue', alpha=0.7)
# plt.xticks(rotation=90)
# plt.xlabel("Features")
# plt.ylabel("Importance Score")
# plt.title("Feature Importance from XGBoost Model")
# plt.grid(True)
# plt.show()

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'XGBoost ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for reference
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#%%
# Applying to a set of variables

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_curve
)

# Define multiple feature sets
feature_sets = {
"X_selected ": ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11","feature13","feature14", "feature16", "feature17", "feature18","feature19", "feature20", "feature21"],

"X_category_one ": ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11"],

"X_feature16" : ["feature16"],

"X_category_two ": ["feature16", "feature18", "feature17", "feature18","feature19"],

"X_category_three ": ["feature13", "feature20", "feature14", "feature9"]
    

    
}

# Define target variable
y = data['outcome'].values  

# Store results
results = []

# Iterate over different feature sets
for name, features in feature_sets.items():
    # Select X variables
    X = data[features].values  
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Apply SMOTE to handle class imbalance
    oversample = SMOTE()
    X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)
    
    # Train XGB Model

    model = XGBClassifier(
       scale_pos_weight=15,
       colsample_bytree=0.8, 
       learning_rate=0.1, 
       max_depth=15, 
       min_child_weight=1, 
       n_estimators=200, 
       subsample=0.6, 
       eval_metric='logloss', 
       random_state=0
   )
    
    # Fit model
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict probabilities for ROC and Precision-Recall AUC
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Set adjustable threshold
    threshold = 0.3  # Change this value to adjust the decision boundary
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Compute Metrics
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Compute Precision-Recall AUC
    precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recalls, precisions)
    
   
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'XGBoost ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for reference
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('XGBoost Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    
    # Plot Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color="blue", label=f"PR Curve (AUC = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for XGBoost (SMOTE)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Compute and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("XGBoost Confusion Matrix")
    plt.show()
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Compute False Positive Rate (FPR)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Store results
    results.append([name, roc_auc, pr_auc, accuracy, precision, recall, fpr])
    

# Convert results to DataFrame
metrics_df = pd.DataFrame(results, columns=["Feature Set", "ROC AUC", "PR AUC", "Accuracy", "Precision", "Recall", "FPR"])


# Display the table
print("XGBoost Performance Metrics:")
print(metrics_df)

# Export to HTML for sharing or viewing in a browser
metrics_df.to_html("XGBoost_metrics.html")
print("\nPerformance metrics have been saved to 'XGBoost_metrics.html'. Open this file in a browser for a clean view.")

#%%
# SVM

#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    precision_recall_curve,
    auc,
)
from imblearn.over_sampling import SVMSMOTE  
from sklearn.svm import SVC


# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 3: Handle class imbalance using SVM-SMOTE
svm_smote = SVMSMOTE(random_state=42)
X_train_resampled, y_train_resampled = svm_smote.fit_resample(X_train, y_train)

# Step 4: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train an SVM model
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train_resampled)

# Step 6: Evaluate the model
y_pred = svm_model.predict(X_test_scaled)
y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Calculate Precision-Recall AUC
precision_values, recall_values, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall_values, precision_values)

# Print results
print("SVM Model Performance:")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"P-R AUC: {pr_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#%% SVM

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    precision_recall_curve,
    auc,
)
from imblearn.over_sampling import SVMSMOTE  
from sklearn.svm import SVC

# Define multiple feature sets
feature_sets = {
"X_selected ": ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11","feature13","feature14", "feature16", "feature17", "feature18","feature19", "feature20", "feature21"],

"X_category_one ": ["feature1", "feature2", "feature3", "feature4", "feature5","feature7", "feature9","feature10", "feature11"],

"X_feature16" : ["feature16"],

"X_category_two ": ["feature16", "feature18", "feature17", "feature18","feature19"],

"X_category_three ": ["feature13", "feature20", "feature14", "feature9"]
    

    
}
# Define target variable
y = data['outcome'].values  

# Store results
results = []

# Iterate over different feature sets
for name, features in feature_sets.items():
    # Select X variables
    X = data[features].values  
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Apply SVMSMOTE to handle class imbalance
    oversample = SVMSMOTE()
    X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)
    
    # train svm model
    model = SVC(probability=True, random_state=42)
        
    # Fit model
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict probabilities for ROC and Precision-Recall AUC
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Set adjustable threshold
    threshold = 0.3  # Change this value to adjust the decision boundary
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Compute Metrics
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Compute Precision-Recall AUC
    precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recalls, precisions)
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Compute False Positive Rate (FPR)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Store results
    results.append([name, roc_auc, pr_auc, accuracy, precision, recall, fpr])
    

# Convert results to DataFrame
metrics_df = pd.DataFrame(results, columns=["Feature Set", "ROC AUC", "PR AUC", "Accuracy", "Precision", "Recall", "FPR"])

# Display the table
print("SVM Performance Metrics:")
print(metrics_df)

# Export to HTML for sharing or viewing in a browser
metrics_df.to_html("SVM_forest_metrics.html")
print("\nPerformance metrics have been saved to 'SVM_metrics.html'. Open this file in a browser for a clean view.")


#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    precision_recall_curve,
    auc,  # For calculating P-R AUC
)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# # Step 1: Load the dataset
# data = pd.read_csv("")

# # Drop irrelevant columns
# data.drop(columns=["Unnamed: 0", "ID"], inplace=True)

# # Features and target variable
# X = data.drop(columns=["outcome"])
# y = data["outcome"]

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 3: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 4: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Step 5: Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}

# Step 6: Train and evaluate models
results = {}
for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train_resampled)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate Precision-Recall AUC
    precision_values, recall_values, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_values, precision_values)
    
    # Store results
    results[name] = {
        "ROC-AUC": roc_auc,
        "P-R AUC": pr_auc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Accuracy": accuracy,
    }

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results).T
print("Model Performance Comparison:")
print(results_df)

# Step 7: Hyperparameter tuning for the best model
best_model_name = results_df["ROC-AUC"].idxmax()
best_model = models[best_model_name]

if best_model_name == "Random Forest":
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    }
elif best_model_name == "Gradient Boosting":
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
    }
elif best_model_name == "XGBoost":
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
    }
else:
    param_grid = {}

# Perform GridSearchCV
if param_grid:
    grid_search = GridSearchCV(best_model, param_grid, scoring="roc_auc", cv=5, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_resampled)
    best_model = grid_search.best_estimator_

# Step 8: Final evaluation of the best model
y_pred_final = best_model.predict(X_test_scaled)
y_pred_proba_final = best_model.predict_proba(X_test_scaled)[:, 1]

# Calculate final metrics
final_roc_auc = roc_auc_score(y_test, y_pred_proba_final)
final_precision = precision_score(y_test, y_pred_final)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)
final_accuracy = accuracy_score(y_test, y_pred_final)

# Calculate final Precision-Recall AUC
precision_values_final, recall_values_final, _ = precision_recall_curve(y_test, y_pred_proba_final)
final_pr_auc = auc(recall_values_final, precision_values_final)

# Print final metrics
print("\nBest Model:", best_model_name)
print("Final Metrics:")
print(f"ROC-AUC: {final_roc_auc:.4f}")
print(f"P-R AUC: {final_pr_auc:.4f}")
print(f"Precision: {final_precision:.4f}")
print(f"Recall: {final_recall:.4f}")
print(f"F1-Score: {final_f1:.4f}")
print(f"Accuracy: {final_accuracy:.4f}")
#%%

