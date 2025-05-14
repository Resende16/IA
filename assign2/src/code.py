
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_curve, auc, 
                            classification_report, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
df = pd.read_csv('employee_promotion_updated.csv')

# =============================================
# 1. Exploratory Data Analysis (EDA)
# =============================================

print("========== Basic Dataset Information ==========")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())

# Target variable distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='is_promoted', data=df)
plt.title('Distribution of Promotion Status (Target Variable)')
plt.xlabel('Promoted (1) vs Not Promoted (0)')
plt.ylabel('Count')
plt.show()

promoted_percentage = df['is_promoted'].value_counts(normalize=True) * 100
print(f"\nPromotion percentage:\n{promoted_percentage}")

# Numerical features analysis
numerical_cols = ['no_of_trainings', 'age', 'previous_year_rating', 
                  'length_of_service', 'avg_training_score']
print("\nNumerical features summary statistics:")
print(df[numerical_cols].describe())

# Plot distributions of numerical features
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Categorical features analysis
categorical_cols = ['department', 'region', 'education', 'gender', 'recruitment_channel', 'awards_won']

plt.figure(figsize=(18, 15))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 2, i)
    if df[col].nunique() > 10:  # For regions with many categories
        sns.countplot(y=col, data=df, order=df[col].value_counts().index[:10])
    else:
        sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation analysis
plt.figure(figsize=(10, 8))
corr_matrix = df[numerical_cols + ['is_promoted']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# =============================================
# 2. Data Preprocessing
# =============================================

# Drop employee_id as it's not a useful feature
df = df.drop('employee_id', axis=1)

# Define features and target
X = df.drop('is_promoted', axis=1)
y = df['is_promoted']

# Split data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define preprocessing for numerical and categorical columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# =============================================
# 3. Model Training and Evaluation
# =============================================

# Define models to evaluate
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'k-NN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Dictionary to store results
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'ROC AUC': [],
    'Training Time (s)': [],
    'Prediction Time (s)': []
}

# Train and evaluate each model
for name, model in models.items():
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train model and measure time
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Make predictions and measure time
    start_time = time.time()
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else [0]*len(y_test)
    predict_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if hasattr(model, 'predict_proba') else 0
    
    # Store results
    results['Model'].append(name)
    results['Accuracy'].append(accuracy)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['F1 Score'].append(f1)
    results['ROC AUC'].append(roc_auc)
    results['Training Time (s)'].append(train_time)
    results['Prediction Time (s)'].append(predict_time)
    
    # Print classification report
    print(f"\n========== {name} ==========")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Promoted', 'Promoted'],
                yticklabels=['Not Promoted', 'Promoted'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # Plot ROC curve if model supports probability predictions
    if hasattr(model, 'predict_proba'):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.show()

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print("\n========== Model Comparison ==========")
print(results_df)

# Plot model comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
plt.figure(figsize=(15, 10))
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    sns.barplot(x=metric, y='Model', data=results_df.sort_values(by=metric, ascending=False))
    plt.title(f'Model Comparison by {metric}')
plt.tight_layout()
plt.show()

# Plot training and prediction times
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='Training Time (s)', y='Model', data=results_df.sort_values(by='Training Time (s)', ascending=False))
plt.title('Training Time Comparison')

plt.subplot(1, 2, 2)
sns.barplot(x='Prediction Time (s)', y='Model', data=results_df.sort_values(by='Prediction Time (s)', ascending=False))
plt.title('Prediction Time Comparison')
plt.tight_layout()
plt.show()

# =============================================
# 4. Hyperparameter Tuning for Best Model
# =============================================

# Based on results, let's tune the Random Forest (usually performs well)
print("\n========== Hyperparameter Tuning for Random Forest ==========")

# Create pipeline for tuning
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# Define parameter grid
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Perform grid search
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, 
                          scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"\nBest parameters: {best_params}")

# Evaluate best model
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nBest model classification report:")
print(classification_report(y_test, y_pred))

print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# Plot feature importance for the best Random Forest model
# Get feature names after one-hot encoding
preprocessor.fit(X_train)
numeric_feature_names = numerical_cols
categorical_feature_names = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
all_feature_names = np.concatenate([numeric_feature_names, categorical_feature_names])

# Get feature importances
importances = best_model.named_steps['model'].feature_importances_

# Create DataFrame for feature importances
feature_importances = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values('Importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', 
            data=feature_importances.head(20))
plt.title('Top 20 Feature Importances from Random Forest')
plt.tight_layout()
plt.show()

# =============================================
# 5. Learning Curves Analysis
# =============================================

# Plot learning curves for the best model
plt.figure(figsize=(10, 6))
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train, y_train, cv=5, 
    scoring='f1', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

plt.title('Learning Curves for Random Forest')
plt.xlabel('Training examples')
plt.ylabel('F1 Score')
plt.legend(loc='best')
plt.grid()
plt.show()

# =============================================
# 6. Final Model Evaluation
# =============================================

# Train final model on entire training data with best parameters
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        random_state=42,
        n_estimators=best_params['model__n_estimators'],
        max_depth=best_params['model__max_depth'],
        min_samples_split=best_params['model__min_samples_split'],
        min_samples_leaf=best_params['model__min_samples_leaf']
    ))
])

final_model.fit(X_train, y_train)

# Evaluate on test set
y_pred_final = final_model.predict(X_test)
y_proba_final = final_model.predict_proba(X_test)[:, 1]

print("\n========== Final Model Evaluation ==========")
print(classification_report(y_test, y_pred_final))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba_final):.4f}")

# Plot final confusion matrix
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Promoted', 'Promoted'],
            yticklabels=['Not Promoted', 'Promoted'])
plt.title('Final Confusion Matrix - Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Plot final ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba_final)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Final ROC Curve - Random Forest')
plt.legend(loc="lower right")
plt.show()