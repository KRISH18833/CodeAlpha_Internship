# =============================================================================
# DISEASE PREDICTION MODEL — CodeAlpha Internship Task 4
# Dataset: Pima Indians Diabetes Dataset
# Objective: Predict the likelihood of diabetes based on medical data
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# =============================================================================
# STEP 1: LOAD DATASET
# =============================================================================
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

# Update path if your file is in a different location
df = pd.read_csv('D:\Code Alpha Internship\Task 2\Diabetes.csv')

print(f"Dataset Shape : {df.shape}")
print(f"\nColumns       : {df.columns.tolist()}")
print(f"\nTarget Distribution:\n{df['Outcome'].value_counts()}")
print(f"\nDiabetes Rate : {df['Outcome'].mean() * 100:.2f}%")

# =============================================================================
# STEP 2: DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Data Preprocessing")
print("=" * 60)

# Columns where 0 is biologically invalid — treat as missing
zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0s with NaN
df[zero_invalid_cols] = df[zero_invalid_cols].replace(0, np.nan)

print("Missing values after replacing invalid zeros:")
print(df.isnull().sum())

# Fill missing values with median (robust to outliers)
for col in zero_invalid_cols:
    df[col].fillna(df[col].median(), inplace=True)

print("\n✅ Missing values filled with column medians")

# Fill any remaining NaNs in the full dataframe
df.fillna(df.median(numeric_only=True), inplace=True)

# =============================================================================
# STEP 3: FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Feature Engineering")
print("=" * 60)

# BMI Category risk score
df['BMI_RISK'] = pd.cut(df['BMI'],
                         bins=[0, 18.5, 24.9, 29.9, 100],
                         labels=[0, 1, 2, 3]).astype(float).fillna(0).astype(int)

# Glucose risk level (clinical thresholds)
df['GLUCOSE_RISK'] = pd.cut(df['Glucose'],
                              bins=[0, 99, 125, 500],
                              labels=[0, 1, 2]).astype(float).fillna(0).astype(int)

# Age risk group
df['AGE_GROUP'] = pd.cut(df['Age'],
                          bins=[0, 30, 45, 60, 100],
                          labels=[0, 1, 2, 3]).astype(float).fillna(0).astype(int)

# Insulin-Glucose interaction
df['INSULIN_GLUCOSE'] = df['Insulin'] * df['Glucose']

print("Engineered Features Added:")
print("  ✅ BMI_RISK         — BMI category risk score (0=underweight to 3=obese)")
print("  ✅ GLUCOSE_RISK     — Glucose risk level based on clinical thresholds")
print("  ✅ AGE_GROUP        — Age group risk category")
print("  ✅ INSULIN_GLUCOSE  — Insulin & Glucose interaction term")

# =============================================================================
# STEP 4: PREPARE FEATURES AND TARGET
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Preparing Features & Target")
print("=" * 60)

X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Training samples  : {X_train.shape[0]}")
print(f"Testing samples   : {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")

# =============================================================================
# STEP 5: TRAIN MODELS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Training Models")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM":                 SVC(kernel='rbf', probability=True, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        "model":     model,
        "y_pred":    y_pred,
        "y_proba":   y_proba,
        "roc_auc":   roc_auc,
        "precision": report['1']['precision'],
        "recall":    report['1']['recall'],
        "f1":        report['1']['f1-score'],
        "accuracy":  report['accuracy'],
    }

    print(f"\n📌 {name}")
    print(f"   Accuracy  : {report['accuracy']:.4f}")
    print(f"   Precision : {report['1']['precision']:.4f}")
    print(f"   Recall    : {report['1']['recall']:.4f}")
    print(f"   F1-Score  : {report['1']['f1-score']:.4f}")
    print(f"   ROC-AUC   : {roc_auc:.4f}")

# =============================================================================
# STEP 6: VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Generating Visualizations")
print("=" * 60)

fig = plt.figure(figsize=(22, 18))
fig.suptitle("Disease Prediction Model — Diabetes Report", fontsize=16, fontweight='bold', y=0.98)

colors = ['#3498db', '#e67e22', '#9b59b6']

# --- Plot 1: Target Distribution ---
ax1 = fig.add_subplot(3, 3, 1)
counts = df['Outcome'].value_counts()
bars = ax1.bar(['No Diabetes\n(0)', 'Diabetes\n(1)'], counts.values,
               color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)
ax1.bar_label(bars, fmt='%d', padding=3, fontsize=10)
ax1.set_title('Target Distribution', fontweight='bold')
ax1.set_ylabel('Count')

# --- Plot 2: Correlation Heatmap ---
ax2 = fig.add_subplot(3, 3, 2)
corr = df[['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure',
           'DiabetesPedigreeFunction', 'GLUCOSE_RISK', 'BMI_RISK', 'Outcome']].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
            ax=ax2, linewidths=0.5, annot_kws={"size": 7})
ax2.set_title('Feature Correlation Heatmap', fontweight='bold')
ax2.tick_params(axis='x', rotation=45, labelsize=7)
ax2.tick_params(axis='y', rotation=0, labelsize=7)

# --- Plot 3: Model Comparison Bar Chart ---
ax3 = fig.add_subplot(3, 3, 3)
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metric_names))
width = 0.25
for i, (name, res) in enumerate(results.items()):
    vals = [res['accuracy'], res['precision'], res['recall'], res['f1'], res['roc_auc']]
    ax3.bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.85)
ax3.set_xticks(x + width)
ax3.set_xticklabels(metric_names, fontsize=8)
ax3.set_ylim(0, 1.1)
ax3.set_title('Model Performance Comparison', fontweight='bold')
ax3.set_ylabel('Score')
ax3.legend(fontsize=7)

# --- Plots 4, 5, 6: Confusion Matrices ---
for i, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(3, 3, 4 + i)
    cm = confusion_matrix(y_test, res['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['No Diabetes', 'Diabetes'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix\n{name}', fontweight='bold', fontsize=9)
    ax.tick_params(labelsize=8)

# --- Plots 7, 8, 9: ROC Curves ---
for i, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(3, 3, 7 + i)
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"AUC = {res['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=8)
    ax.set_ylabel('True Positive Rate', fontsize=8)
    ax.set_title(f'ROC Curve\n{name}', fontweight='bold', fontsize=9)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('disease_prediction_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot saved as 'disease_prediction_results.png'")

# =============================================================================
# STEP 7: FEATURE IMPORTANCE (Random Forest)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Feature Importance (Random Forest)")
print("=" * 60)

rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Feature Importances — Random Forest', fontweight='bold')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('disease_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Feature importance plot saved as 'disease_feature_importance.png'")

print("\nFeature Importances:")
for feat, score in importances.items():
    print(f"  {feat:<28} : {score:.4f}")

# =============================================================================
# STEP 8: FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: FINAL SUMMARY")
print("=" * 60)

best_model = max(results, key=lambda k: results[k]['roc_auc'])
print(f"\n🏆 Best Model: {best_model}")
print(f"   ROC-AUC  : {results[best_model]['roc_auc']:.4f}")
print(f"   Accuracy : {results[best_model]['accuracy']:.4f}")
print(f"   F1-Score : {results[best_model]['f1']:.4f}")

print("\n📊 All Models Summary:")
print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
print("-" * 72)
for name, res in results.items():
    print(f"{name:<22} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
          f"{res['recall']:>10.4f} {res['f1']:>10.4f} {res['roc_auc']:>10.4f}")

print("\n✅ Disease Prediction Model Complete!")
print("   Files generated:")
print("   → disease_prediction_results.png")
print("   → disease_feature_importance.png")