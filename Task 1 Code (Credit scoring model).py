# =============================================================================
# CREDIT SCORING MODEL — CodeAlpha Internship Task 1
# Dataset: Default of Credit Card Clients (UCI)
# Objective: Predict creditworthiness using past financial data
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
from sklearn.tree import DecisionTreeClassifier
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

# Update the path below if your file is in a different location
df = pd.read_excel(r'D:\Code Alpha Internship\Task 1\CreditCardDataset.xls', header=1)

# Drop ID column (not useful for prediction)
df.drop(columns=['ID'], inplace=True)

# Rename target column for convenience
df.rename(columns={'default payment next month': 'DEFAULT'}, inplace=True)

print(f"Dataset Shape: {df.shape}")
print(f"\nTarget Distribution:\n{df['DEFAULT'].value_counts()}")
print(f"\nDefault Rate: {df['DEFAULT'].mean() * 100:.2f}%")

# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Feature Engineering")
print("=" * 60)

# --- Engineered Features ---

# Average payment delay over 6 months (higher = worse credit behavior)
pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
df['AVG_PAY_DELAY'] = df[pay_cols].mean(axis=1)

# Total bill amount over 6 months
bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
df['TOTAL_BILL'] = df[bill_cols].sum(axis=1)

# Total payment made over 6 months
pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
df['TOTAL_PAID'] = df[pay_amt_cols].sum(axis=1)

# Payment ratio: how much of the bill was actually paid (higher = better)
df['PAY_RATIO'] = df['TOTAL_PAID'] / (df['TOTAL_BILL'] + 1)  # +1 to avoid division by zero

# Credit utilization ratio: how much of the credit limit is used
df['UTIL_RATIO'] = df['TOTAL_BILL'] / (df['LIMIT_BAL'] + 1)

print("Engineered Features Added:")
print("  ✅ AVG_PAY_DELAY  — average payment delay across 6 months")
print("  ✅ TOTAL_BILL     — total billed amount across 6 months")
print("  ✅ TOTAL_PAID     — total amount paid across 6 months")
print("  ✅ PAY_RATIO      — ratio of payments made to bills")
print("  ✅ UTIL_RATIO     — credit utilization ratio")

# =============================================================================
# STEP 3: PREPARE FEATURES AND TARGET
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Preparing Features & Target")
print("=" * 60)

X = df.drop(columns=['DEFAULT'])
y = df['DEFAULT']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")

# =============================================================================
# STEP 4: TRAIN MODELS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Training Models")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

results = {}

for name, model in models.items():
    # Logistic Regression needs scaled data
    X_tr = X_train_scaled if name == "Logistic Regression" else X_train
    X_te = X_test_scaled  if name == "Logistic Regression" else X_test

    model.fit(X_tr, y_train)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        "model":    model,
        "y_pred":   y_pred,
        "y_proba":  y_proba,
        "roc_auc":  roc_auc,
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
# STEP 5: VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Generating Visualizations")
print("=" * 60)

fig = plt.figure(figsize=(22, 18))
fig.suptitle("Credit Scoring Model — Full Report", fontsize=16, fontweight='bold', y=0.98)

# --- Plot 1: Target Distribution ---
ax1 = fig.add_subplot(3, 3, 1)
counts = df['DEFAULT'].value_counts()
bars = ax1.bar(['No Default\n(0)', 'Default\n(1)'], counts.values,
               color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)
ax1.bar_label(bars, fmt='%d', padding=3, fontsize=10)
ax1.set_title('Target Distribution', fontweight='bold')
ax1.set_ylabel('Count')

# --- Plot 2: Correlation Heatmap (top features) ---
ax2 = fig.add_subplot(3, 3, 2)
top_features = ['LIMIT_BAL', 'AGE', 'AVG_PAY_DELAY', 'PAY_RATIO',
                'UTIL_RATIO', 'TOTAL_BILL', 'TOTAL_PAID', 'DEFAULT']
corr = df[top_features].corr()
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
colors = ['#3498db', '#e67e22', '#9b59b6']
for i, (name, res) in enumerate(results.items()):
    vals = [res['accuracy'], res['precision'], res['recall'], res['f1'], res['roc_auc']]
    bars = ax3.bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.85)
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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Default', 'Default'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix\n{name}', fontweight='bold', fontsize=9)
    ax.tick_params(labelsize=8)

# --- Plots 7, 8, 9: ROC Curves ---
for i, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(3, 3, 7 + i)
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax.plot(fpr, tpr, color=colors[i], lw=2,
            label=f"AUC = {res['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=8)
    ax.set_ylabel('True Positive Rate', fontsize=8)
    ax.set_title(f'ROC Curve\n{name}', fontweight='bold', fontsize=9)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('credit_scoring_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot saved as 'credit_scoring_results.png'")

# =============================================================================
# STEP 6: FEATURE IMPORTANCE (Random Forest)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Feature Importance (Random Forest)")
print("=" * 60)

rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Top 15 Feature Importances — Random Forest', fontweight='bold')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Feature importance plot saved as 'feature_importance.png'")

print("\nTop 10 Most Important Features:")
for feat, score in importances.head(10).items():
    print(f"  {feat:<20} : {score:.4f}")

# =============================================================================
# STEP 7: FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: FINAL SUMMARY")
print("=" * 60)

best_model = max(results, key=lambda k: results[k]['roc_auc'])
print(f"\n🏆 Best Model: {best_model}")
print(f"   ROC-AUC  : {results[best_model]['roc_auc']:.4f}")
print(f"   Accuracy : {results[best_model]['accuracy']:.4f}")
print(f"   F1-Score : {results[best_model]['f1']:.4f}")

print("\n📊 All Models Summary:")
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
print("-" * 75)
for name, res in results.items():
    print(f"{name:<25} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
          f"{res['recall']:>10.4f} {res['f1']:>10.4f} {res['roc_auc']:>10.4f}")

print("\n✅ Credit Scoring Model Complete!")
print("   Files generated:")
print("   → credit_scoring_results.png")
print("   → feature_importance.png")