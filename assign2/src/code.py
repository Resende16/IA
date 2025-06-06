import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import time
import warnings
import plotext as pltx
from rich.console import Console
console = Console()

warnings.filterwarnings('ignore')

np.random.seed(42)

df = pd.read_csv('employee_promotion_updated.csv')

plt.figure(figsize=(8, 5))
sns.countplot(x='is_promoted', data=df)
plt.title('Distribution of Promotion Status (Target Variable)')
plt.xlabel('Promoted (1) vs Not Promoted (0)')
plt.ylabel('Count')
plt.show()

promoted_percentage = df['is_promoted'].value_counts(normalize=True) * 100
console.print(f"\n========== Promotion percentage ==========\n", style="bold blue")
print(f"Promoted: {promoted_percentage[1]:.2f}%")
print(f"Not Promoted: {promoted_percentage[0]:.2f}%\n")

numerical_cols = ['no_of_trainings', 'age', 'previous_year_rating', 
                  'length_of_service', 'avg_training_score']

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

categorical_cols = ['department', 'region', 'education', 'gender', 'recruitment_channel', 'awards_won']

plt.figure(figsize=(18, 15))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 2, i)
    if df[col].nunique() > 10:  
        sns.countplot(y=col, data=df, order=df[col].value_counts().index[:10])
    else:
        sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
corr_matrix = df[numerical_cols + ['is_promoted']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()


df = df.drop('employee_id', axis=1)

X = df.drop('is_promoted', axis=1)
y = df['is_promoted']

# Stratify é necessário porque o dataset está desiquilibrado (apenas 9% promoted)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


#Pré-Processamento -> Definimos o tratamento de dados e como as variaveis vao ser transformadas!!

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


models = {
    # Decision Tree que vai tomar decisoes binarias (if/else) em cada feature
    'Decision Tree': DecisionTreeClassifier(random_state=42),

    # k-NN classifica com base nos k mais proximo no dataset, por isso é que demora mais (quantos mais dados mais lento é)
    #Neste caso muitos dados por isso fica muito lento (cerca de 30 segundos)
    'k-NN': KNeighborsClassifier(),

    # Modelo linear e com base em probabilidades
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [], # Dos que o modelo promoveu quantos foram mesmo promovidos (precisão)
    'Recall': [], # Dos que foram mesmo promovidos quantos o modelo identificou (sensibilidade)
    # O Recall vai ser muito importante para evitar falsos negativos no fundo
    'F1 Score': [], # Tem em conta precision e recall (importante para estes datasets desiquilibrados)
    'ROC AUC': [],
    'Training Time (s)': [],
    'Prediction Time (s)': []
}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else [0]*len(y_test)
    predict_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if hasattr(model, 'predict_proba') else 0
    
    results['Model'].append(name)
    results['Accuracy'].append(accuracy)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['F1 Score'].append(f1)
    results['ROC AUC'].append(roc_auc)
    results['Training Time (s)'].append(train_time)
    results['Prediction Time (s)'].append(predict_time)
    
    console.print(f"\n========== {name} ==========", style="bold blue")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Promoted', 'Promoted'],
                yticklabels=['Not Promoted', 'Promoted'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
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

results_df = pd.DataFrame(results)
console.print("\n========== Model Comparison ==========", style="bold blue")
print(results_df)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
plt.figure(figsize=(15, 10))
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    sns.barplot(x=metric, y='Model', data=results_df.sort_values(by=metric, ascending=False))
    plt.title(f'Model Comparison by {metric}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='Training Time (s)', y='Model', data=results_df.sort_values(by='Training Time (s)', ascending=False))
plt.title('Training Time Comparison')

plt.subplot(1, 2, 2)
sns.barplot(x='Prediction Time (s)', y='Model', data=results_df.sort_values(by='Prediction Time (s)', ascending=False))
plt.title('Prediction Time Comparison')
plt.tight_layout()
plt.show()

console.print("\n========== Some Comparison Graphics ==========", style="bold blue")

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']

for metric in metrics_to_plot:
    pltx.clear_figure()
    pltx.title(f"Model Comparison - {metric}")
    pltx.bar(results_df['Model'].tolist(), results_df[metric].tolist())
    pltx.xlabel("Models")
    pltx.ylabel(metric)
    pltx.plotsize(100, 20)
    pltx.show()
    print("\n\n")
