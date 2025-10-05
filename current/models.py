import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    auc, accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# =========================
# Global plot settings
# =========================
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# =========================
# Load data
# =========================
df = pd.read_csv("diving_synthetic_profiles.csv")

print("="*80)
print("COMPREHENSIVE DIVING INCIDENT ANALYSIS")
print("="*80)
print(f"\nDataset: {len(df)} dives")
print(f"Incident rate: {df['Incident'].mean():.2%}")
print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =========================
# Feature Engineering
# =========================
df['MaxDepth_m_sq'] = df['MaxDepth_m'] ** 2
df['Nitrox'] = (df['GasMix'] == 'Nitrox32').astype(int)
df['Depth_x_Time'] = df['MaxDepth_m'] * df['BottomTime_min']

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
summary_stats = df[['MaxDepth_m', 'BottomTime_min', 'AscentRate_m_per_min', 
                    'SafetyStop', 'NDL_Violation', 'Nitrox', 'Incident']].describe()
print(summary_stats)

print("\nIncident rates by key factors:")
print(f"  With Safety Stop: {df[df['SafetyStop']==1]['Incident'].mean():.2%}")
print(f"  Without Safety Stop: {df[df['SafetyStop']==0]['Incident'].mean():.2%}")
print(f"  With NDL Violation: {df[df['NDL_Violation']==1]['Incident'].mean():.2%}")
print(f"  Without NDL Violation: {df[df['NDL_Violation']==0]['Incident'].mean():.2%}")
print(f"  Using Nitrox: {df[df['Nitrox']==1]['Incident'].mean():.2%}")
print(f"  Using Air: {df[df['Nitrox']==0]['Incident'].mean():.2%}")

# =========================
# Multicollinearity (VIF)
# =========================
print("\n" + "="*80)
print("MULTICOLLINEARITY CHECK (Variance Inflation Factors)")
print("="*80)

X_vif = df[['MaxDepth_m', 'MaxDepth_m_sq', 'BottomTime_min', 'AscentRate_m_per_min', 
            'SafetyStop', 'NDL_Violation', 'Depth_x_Time', 'Nitrox']].copy()
X_vif = X_vif.assign(const=1.0)  # add constant for VIF stability

vif_data = pd.DataFrame({
    "Variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
}).sort_values('VIF', ascending=False)

print(vif_data.to_string(index=False))
print("\nNote: VIF > 10 suggests problematic multicollinearity")
print(f"Max VIF: {vif_data['VIF'].max():.2f}")

# Drop the added const column (not used by formulas below)
vif_data = vif_data[vif_data['Variable'] != 'const']

# =========================
# Model Estimation
# =========================
print("\n" + "="*80)
print("MODEL 1: BASELINE LOGIT")
print("="*80)
logit_baseline = smf.logit(
    "Incident ~ MaxDepth_m + BottomTime_min + AscentRate_m_per_min + SafetyStop + NDL_Violation",
    data=df
).fit(disp=0)
print(logit_baseline.summary())

print("\n" + "="*80)
print("MODEL 2: FULL LOGIT MODEL")
print("="*80)
logit_full = smf.logit(
    """Incident ~ MaxDepth_m + I(MaxDepth_m**2) + BottomTime_min + 
       AscentRate_m_per_min + SafetyStop + NDL_Violation + 
       MaxDepth_m:BottomTime_min + Nitrox""",
    data=df
).fit(disp=0)
print(logit_full.summary())

print("\n" + "="*80)
print("MODEL 3: FULL PROBIT MODEL")
print("="*80)
probit_full = smf.probit(
    """Incident ~ MaxDepth_m + I(MaxDepth_m**2) + BottomTime_min + 
       AscentRate_m_per_min + SafetyStop + NDL_Violation + 
       MaxDepth_m:BottomTime_min + Nitrox""",
    data=df
).fit(disp=0)
print(probit_full.summary())

print("\n" + "="*80)
print("MODEL 4: FULL LOGIT WITH ROBUST STANDARD ERRORS (HC3)")
print("="*80)
logit_robust = smf.logit(
    """Incident ~ MaxDepth_m + I(MaxDepth_m**2) + BottomTime_min + 
       AscentRate_m_per_min + SafetyStop + NDL_Violation + 
       MaxDepth_m:BottomTime_min + Nitrox""",
    data=df
).fit(cov_type='HC3', disp=0)
print(logit_robust.summary())

# =========================
# Marginal Effects (Logit Full)
# =========================
print("\n" + "="*80)
print("AVERAGE MARGINAL EFFECTS (Full Logit Model)")
print("="*80)
marginal_effects = logit_full.get_margeff(at='mean')
print(marginal_effects.summary())

me_df = pd.DataFrame({
    'Variable': marginal_effects.margeff_names,
    'Marginal Effect': marginal_effects.margeff,
    'Std Error': marginal_effects.margeff_se,
    'P-value': marginal_effects.pvalues
})

print("\nInterpretation:")
for _, row in me_df.iterrows():
    if row['P-value'] < 0.05:
        print(f"  {row['Variable']}: {row['Marginal Effect']:.4f} "
              f"(~{row['Marginal Effect']*100:.2f} percentage points per unit)")

# =========================
# Predictions & Metrics
# =========================
print("\n" + "="*80)
print("MODEL PERFORMANCE METRICS")
print("="*80)

df['pred_prob_baseline'] = logit_baseline.predict()
df['pred_prob_full'] = logit_full.predict()
df['pred_prob_probit'] = probit_full.predict()

threshold = 0.5
df['pred_class_baseline'] = (df['pred_prob_baseline'] > threshold).astype(int)
df['pred_class_full'] = (df['pred_prob_full'] > threshold).astype(int)

models_eval = {
    'Baseline Logit': (df['pred_class_baseline'], df['pred_prob_baseline']),
    'Full Logit': (df['pred_class_full'], df['pred_prob_full']),
    'Full Probit': ((df['pred_prob_probit'] > threshold).astype(int), df['pred_prob_probit'])
}

performance_metrics = []
for model_name, (pred_class, pred_prob) in models_eval.items():
    accuracy = accuracy_score(df['Incident'], pred_class)
    precision = precision_score(df['Incident'], pred_class, zero_division=0)
    recall = recall_score(df['Incident'], pred_class, zero_division=0)
    f1 = f1_score(df['Incident'], pred_class, zero_division=0)
    roc_auc = roc_auc_score(df['Incident'], pred_prob)
    performance_metrics.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })

perf_df = pd.DataFrame(performance_metrics)
print(perf_df.to_string(index=False))

# =========================
# Confusion Matrices
# =========================
print("\n" + "="*80)
print("CONFUSION MATRICES")
print("="*80)
for model_name, (pred_class, _) in models_eval.items():
    cm = confusion_matrix(df['Incident'], pred_class)
    print(f"\n{model_name}:")
    print(f"                 Predicted No  Predicted Yes")
    print(f"  Actual No      {cm[0,0]:8d}      {cm[0,1]:8d}")
    print(f"  Actual Yes     {cm[1,0]:8d}      {cm[1,1]:8d}")
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  Sensitivity (Recall): {sensitivity:.3f}")
    print(f"  Specificity: {specificity:.3f}")

# =========================
# Hosmer–Lemeshow GOF
# =========================
print("\n" + "="*80)
print("HOSMER-LEMESHOW GOODNESS-OF-FIT TEST (Full Logit)")
print("="*80)

def hosmer_lemeshow_test(y_true, y_pred, g=10):
    df_hl = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df_hl['decile'] = pd.qcut(df_hl['y_pred'], g, duplicates='drop')
    hl_table = df_hl.groupby('decile').agg(
        observed=('y_true', 'sum'),
        n=('y_true', 'count'),
        expected=('y_pred', 'sum')
    )
    observed = hl_table['observed'].values
    expected = hl_table['expected'].values
    n = hl_table['n'].values
    eps = 1e-10
    chi_sq = np.sum((observed - expected)**2 / (expected * (1 - expected/np.maximum(n, eps)) + eps))
    df_test = g - 2
    p_value = 1 - stats.chi2.cdf(chi_sq, df_test)
    return chi_sq, df_test, p_value

hl_chi2, hl_df, hl_pval = hosmer_lemeshow_test(df['Incident'], df['pred_prob_full'])
print(f"Chi-square statistic: {hl_chi2:.4f}")
print(f"Degrees of freedom: {hl_df}")
print(f"P-value: {hl_pval:.4f}")
print(f"Result: {'Good fit' if hl_pval > 0.05 else 'Poor fit'} (p > 0.05 indicates good fit)")

# =========================
# Cross-Validation
# =========================
print("\n" + "="*80)
print("CROSS-VALIDATION (5-Fold)")
print("="*80)

X = df[['MaxDepth_m', 'MaxDepth_m_sq', 'BottomTime_min', 'AscentRate_m_per_min',
        'SafetyStop', 'NDL_Violation', 'Depth_x_Time', 'Nitrox']]
y = df['Incident']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=1000, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')

print(f"ROC-AUC scores across folds: {cv_scores}")
print(f"Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# =========================
# Model Comparison Table
# =========================
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

models = {
    'Baseline Logit': logit_baseline,
    'Full Logit': logit_full,
    'Full Probit': probit_full,
    'Full Logit (Robust SE)': logit_robust
}

comparison = pd.DataFrame({
    'Model': list(models.keys()),
    'Log-Likelihood': [m.llf for m in models.values()],
    'AIC': [m.aic for m in models.values()],
    'BIC': [m.bic for m in models.values()],
    'Pseudo R²': [m.prsquared for m in models.values()],
    'N Parameters': [len(m.params) for m in models.values()]
})

print(comparison.to_string(index=False))

# =========================
# Likelihood Ratio Test
# =========================
print("\n" + "="*80)
print("LIKELIHOOD RATIO TESTS")
print("="*80)

def lr_test(model_restricted, model_full):
    lr_stat = 2 * (model_full.llf - model_restricted.llf)
    df_lr = len(model_full.params) - len(model_restricted.params)
    p_value = stats.chi2.sf(lr_stat, df_lr)
    return lr_stat, df_lr, p_value

lr_full, df_full, p_full = lr_test(logit_baseline, logit_full)
print(f"\nFull model vs Baseline:")
print(f"  LR statistic: {lr_full:.4f}")
print(f"  df: {df_full}")
print(f"  p-value: {p_full:.6f}")
print(f"  Result: {'Reject null - Full model is better' if p_full < 0.05 else 'Fail to reject - Models similar'}")

# =========================
# Odds Ratios
# =========================
print("\n" + "="*80)
print("ODDS RATIOS (Full Logit Model)")
print("="*80)

ci = logit_full.conf_int()
odds_ratios = pd.DataFrame({
    'Variable': logit_full.params.index,
    'Coefficient': logit_full.params.values,
    'Odds Ratio': np.exp(logit_full.params.values),
    'CI_Lower': np.exp(ci[0].values),
    'CI_Upper': np.exp(ci[1].values),
    'P-value': logit_full.pvalues.values
})
print(odds_ratios.to_string(index=False))

# =========================
# Visualizations
# =========================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(18, 12))

# 1. ROC Curves
ax1 = plt.subplot(2, 3, 1)
for model_name, (_, pred_prob) in models_eval.items():
    fpr, tpr, _ = roc_curve(df['Incident'], pred_prob)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)

ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves Comparison', fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(alpha=0.3)

# 2. Predicted Probability vs Depth
ax2 = plt.subplot(2, 3, 2)
depth_range = np.linspace(df['MaxDepth_m'].min(), df['MaxDepth_m'].max(), 100)
pred_data = pd.DataFrame({
    'MaxDepth_m': depth_range,
    'MaxDepth_m_sq': depth_range**2,
    'BottomTime_min': df['BottomTime_min'].mean(),
    'AscentRate_m_per_min': df['AscentRate_m_per_min'].mean(),
    'SafetyStop': 1,
    'NDL_Violation': 0,
    'Depth_x_Time': depth_range * df['BottomTime_min'].mean(),
    'Nitrox': 0
})
pred_data['I(MaxDepth_m ** 2)'] = pred_data['MaxDepth_m_sq']
pred_data['MaxDepth_m:BottomTime_min'] = pred_data['Depth_x_Time']
pred_probs = logit_full.predict(pred_data)
ax2.plot(depth_range, pred_probs, linewidth=2)
ax2.fill_between(depth_range, 0, pred_probs, alpha=0.3)
ax2.set_xlabel('Maximum Depth (m)')
ax2.set_ylabel('Predicted Incident Probability')
ax2.set_title('Risk vs Depth (means; safety stop)', fontweight='bold')
ax2.grid(alpha=0.3)

# 3. Predicted Probability vs Bottom Time
ax3 = plt.subplot(2, 3, 3)
time_range = np.linspace(df['BottomTime_min'].min(), df['BottomTime_min'].max(), 100)
mean_depth = df['MaxDepth_m'].mean()
pred_data_time = pd.DataFrame({
    'MaxDepth_m': mean_depth,
    'MaxDepth_m_sq': mean_depth**2,
    'BottomTime_min': time_range,
    'AscentRate_m_per_min': df['AscentRate_m_per_min'].mean(),
    'SafetyStop': 1,
    'NDL_Violation': 0,
    'Depth_x_Time': mean_depth * time_range,
    'Nitrox': 0
})
pred_data_time['I(MaxDepth_m ** 2)'] = pred_data_time['MaxDepth_m_sq']
pred_data_time['MaxDepth_m:BottomTime_min'] = pred_data_time['Depth_x_Time']
pred_probs_time = logit_full.predict(pred_data_time)
ax3.plot(time_range, pred_probs_time, linewidth=2)
ax3.fill_between(time_range, 0, pred_probs_time, alpha=0.3)
ax3.set_xlabel('Bottom Time (min)')
ax3.set_ylabel('Predicted Incident Probability')
ax3.set_title('Risk vs Bottom Time (mean depth; safety stop)', fontweight='bold')
ax3.grid(alpha=0.3)

# 4. Interaction Effect: Depth × Bottom Time
ax4 = plt.subplot(2, 3, 4)
depths_to_plot = [15, 25, 35, 45]
for depth_val in depths_to_plot:
    time_range_int = np.linspace(10, 70, 100)
    pred_data_int = pd.DataFrame({
        'MaxDepth_m': depth_val,
        'MaxDepth_m_sq': depth_val**2,
        'BottomTime_min': time_range_int,
        'AscentRate_m_per_min': df['AscentRate_m_per_min'].mean(),
        'SafetyStop': 1,
        'NDL_Violation': 0,
        'Depth_x_Time': depth_val * time_range_int,
        'Nitrox': 0
    })
    pred_data_int['I(MaxDepth_m ** 2)'] = pred_data_int['MaxDepth_m_sq']
    pred_data_int['MaxDepth_m:BottomTime_min'] = pred_data_int['Depth_x_Time']
    pred_probs_int = logit_full.predict(pred_data_int)
    ax4.plot(time_range_int, pred_probs_int, linewidth=2, label=f'{depth_val} m')

ax4.set_xlabel('Bottom Time (min)')
ax4.set_ylabel('Predicted Incident Probability')
ax4.set_title('Depth × Bottom Time Interaction', fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# 5. Safety Stop Effect by Depth
ax5 = plt.subplot(2, 3, 5)
depth_range_ss = np.linspace(df['MaxDepth_m'].min(), df['MaxDepth_m'].max(), 50)
for ss_val, ss_label in [(0, 'No Safety Stop'), (1, 'With Safety Stop')]:
    pred_data_ss = pd.DataFrame({
        'MaxDepth_m': depth_range_ss,
        'MaxDepth_m_sq': depth_range_ss**2,
        'BottomTime_min': df['BottomTime_min'].mean(),
        'AscentRate_m_per_min': df['AscentRate_m_per_min'].mean(),
        'SafetyStop': ss_val,
        'NDL_Violation': 0,
        'Depth_x_Time': depth_range_ss * df['BottomTime_min'].mean(),
        'Nitrox': 0
    })
    pred_data_ss['I(MaxDepth_m ** 2)'] = pred_data_ss['MaxDepth_m_sq']
    pred_data_ss['MaxDepth_m:BottomTime_min'] = pred_data_ss['Depth_x_Time']
    pred_probs_ss = logit_full.predict(pred_data_ss)
    ax5.plot(depth_range_ss, pred_probs_ss, linewidth=2, label=ss_label)

ax5.set_xlabel('Maximum Depth (m)')
ax5.set_ylabel('Predicted Incident Probability')
ax5.set_title('Safety Stop Effect by Depth', fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)

# 6. Confusion Matrix Heatmap (Full Logit)
ax6 = plt.subplot(2, 3, 6)
cm = confusion_matrix(df['Incident'], df['pred_class_full'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
            xticklabels=['Predicted No', 'Predicted Yes'],
            yticklabels=['Actual No', 'Actual Yes'])
ax6.set_title('Confusion Matrix (Full Logit)', fontweight='bold')

plt.tight_layout()
plt.savefig('comprehensive_analysis_plots.png', dpi=300, bbox_inches='tight')
print("✓ Plots saved to: comprehensive_analysis_plots.png")
plt.close()

# Partial dependence plots
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
continuous_vars = [
    ('MaxDepth_m', 'Maximum Depth (m)'),
    ('BottomTime_min', 'Bottom Time (min)'),
    ('AscentRate_m_per_min', 'Ascent Rate (m/min)'),
]
for idx, (var, label) in enumerate(continuous_vars):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    var_range = np.linspace(df[var].min(), df[var].max(), 100)
    if var == 'MaxDepth_m':
        pred_df = pd.DataFrame({
            'MaxDepth_m': var_range,
            'MaxDepth_m_sq': var_range**2,
            'BottomTime_min': df['BottomTime_min'].mean(),
            'AscentRate_m_per_min': df['AscentRate_m_per_min'].mean(),
            'SafetyStop': df['SafetyStop'].mean(),
            'NDL_Violation': 0,
            'Depth_x_Time': var_range * df['BottomTime_min'].mean(),
            'Nitrox': 0
        })
    elif var == 'BottomTime_min':
        pred_df = pd.DataFrame({
            'MaxDepth_m': df['MaxDepth_m'].mean(),
            'MaxDepth_m_sq': df['MaxDepth_m'].mean()**2,
            'BottomTime_min': var_range,
            'AscentRate_m_per_min': df['AscentRate_m_per_min'].mean(),
            'SafetyStop': df['SafetyStop'].mean(),
            'NDL_Violation': 0,
            'Depth_x_Time': df['MaxDepth_m'].mean() * var_range,
            'Nitrox': 0
        })
    else:
        pred_df = pd.DataFrame({
            'MaxDepth_m': df['MaxDepth_m'].mean(),
            'MaxDepth_m_sq': df['MaxDepth_m'].mean()**2,
            'BottomTime_min': df['BottomTime_min'].mean(),
            'AscentRate_m_per_min': var_range,
            'SafetyStop': df['SafetyStop'].mean(),
            'NDL_Violation': 0,
            'Depth_x_Time': df['MaxDepth_m'].mean() * df['BottomTime_min'].mean(),
            'Nitrox': 0
        })
    pred_df['I(MaxDepth_m ** 2)'] = pred_df['MaxDepth_m_sq']
    pred_df['MaxDepth_m:BottomTime_min'] = pred_df['Depth_x_Time']
    pred_probs_pd = logit_full.predict(pred_df)
    ax.plot(var_range, pred_probs_pd, linewidth=2.5)
    ax.fill_between(var_range, 0, pred_probs_pd, alpha=0.3)
    ax.set_xlabel(label)
    ax.set_ylabel('Predicted Incident Probability')
    ax.set_title(f'Partial Dependence: {label}', fontweight='bold')
    ax.grid(alpha=0.3)

# NDL violation effect bar
ax = axes[1, 1]
categories = ['No NDL Violation', 'NDL Violation']
probs_ndl = []
for ndl_val in [0, 1]:
    pred_df_ndl = pd.DataFrame({
        'MaxDepth_m': [df['MaxDepth_m'].mean()],
        'MaxDepth_m_sq': [df['MaxDepth_m'].mean()**2],
        'BottomTime_min': [df['BottomTime_min'].mean()],
        'AscentRate_m_per_min': [df['AscentRate_m_per_min'].mean()],
        'SafetyStop': [1],
        'NDL_Violation': [ndl_val],
        'Depth_x_Time': [df['MaxDepth_m'].mean() * df['BottomTime_min'].mean()],
        'Nitrox': [0]
    })
    pred_df_ndl['I(MaxDepth_m ** 2)'] = pred_df_ndl['MaxDepth_m_sq']
    pred_df_ndl['MaxDepth_m:BottomTime_min'] = pred_df_ndl['Depth_x_Time']
    probs_ndl.append(float(logit_full.predict(pred_df_ndl)[0]))

bars = ax.bar(categories, probs_ndl, color=['#6ab04c', '#eb4d4b'], alpha=0.8, edgecolor='black')
ax.set_ylabel('Predicted Incident Probability')
ax.set_title('NDL Violation Effect', fontweight='bold')
ax.grid(alpha=0.3, axis='y')
for bar, prob in zip(bars, probs_ndl):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('partial_dependence_plots.png', dpi=300, bbox_inches='tight')
print("✓ Partial dependence plots saved to: partial_dependence_plots.png")
plt.close()

# =========================
# Residual Analysis
# =========================
print("\n" + "="*80)
print("RESIDUAL ANALYSIS")
print("="*80)

pearson_resid = logit_full.resid_pearson
deviance_resid = logit_full.resid_deviance

print("Pearson residuals summary:")
print(f"  Mean: {pearson_resid.mean():.4f}")
print(f"  Std Dev: {pearson_resid.std():.4f}")
print(f"  Min: {pearson_resid.min():.4f}")
print(f"  Max: {pearson_resid.max():.4f}")

outlier_threshold = 3
outliers = np.abs(pearson_resid) > outlier_threshold
print(f"\nObservations with |Pearson residual| > {outlier_threshold}: {outliers.sum()}")
if outliers.sum() > 0:
    print(f"Outlier dive IDs (first 10): {df.loc[outliers, 'DiveID'].values[:10]}")

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
ax = axes3[0, 0]
ax.scatter(df['pred_prob_full'], pearson_resid, alpha=0.5, s=20)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Fitted Probability')
ax.set_ylabel('Pearson Residuals')
ax.set_title('Residuals vs Fitted Values', fontweight='bold')
ax.grid(alpha=0.3)

ax = axes3[0, 1]
ax.scatter(df['MaxDepth_m'], pearson_resid, alpha=0.5, s=20, c='coral')
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Maximum Depth (m)')
ax.set_ylabel('Pearson Residuals')
ax.set_title('Residuals vs Depth', fontweight='bold')
ax.grid(alpha=0.3)

ax = axes3[1, 0]
ax.scatter(df['BottomTime_min'], pearson_resid, alpha=0.5, s=20, c='steelblue')
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Bottom Time (min)')
ax.set_ylabel('Pearson Residuals')
ax.set_title('Residuals vs Bottom Time', fontweight='bold')
ax.grid(alpha=0.3)

ax = axes3[1, 1]
stats.probplot(pearson_resid, dist="norm", plot=ax)
ax.set_title('Q-Q Plot of Pearson Residuals', fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('residual_plots.png', dpi=300, bbox_inches='tight')
print("✓ Residual plots saved to: residual_plots.png")
plt.close()

# =========================
# Interpretation (Key Findings)
# =========================
print("\n" + "="*80)
print("KEY FINDINGS & INTERPRETATION")
print("="*80)

depth_coef = logit_full.params['MaxDepth_m']
depth_sq_coef = logit_full.params['I(MaxDepth_m ** 2)']
bt_coef = logit_full.params['BottomTime_min']
ascent_coef = logit_full.params['AscentRate_m_per_min']
ss_coef = logit_full.params['SafetyStop']
nitrox_coef = logit_full.params['Nitrox']
ndl_coef = logit_full.params['NDL_Violation']
interaction_coef = logit_full.params['MaxDepth_m:BottomTime_min']

depth_me = me_df.loc[me_df['Variable'] == 'MaxDepth_m', 'Marginal Effect'].values[0]
bt_me = me_df.loc[me_df['Variable'] == 'BottomTime_min', 'Marginal Effect'].values[0]
ascent_me = me_df.loc[me_df['Variable'] == 'AscentRate_m_per_min', 'Marginal Effect'].values[0]
ss_me = me_df.loc[me_df['Variable'] == 'SafetyStop', 'Marginal Effect'].values[0]
nitrox_me = me_df.loc[me_df['Variable'] == 'Nitrox', 'Marginal Effect'].values[0]
ndl_me = me_df.loc[me_df['Variable'] == 'NDL_Violation', 'Marginal Effect'].values[0]
interaction_me = me_df.loc[me_df['Variable'] == 'MaxDepth_m:BottomTime_min', 'Marginal Effect'].values[0]

print("\n1. DEPTH EFFECT:")
print(f"   Linear coefficient: {depth_coef:.4f}")
print(f"   Quadratic coefficient: {depth_sq_coef:.6f}")
print(f"   Marginal effect: {depth_me:.4f} ({depth_me*100:.2f} pp per meter)")
print("   → " + ("Risk ACCELERATES at greater depths (convex)" if depth_sq_coef > 0 else
               "Risk increases with depth but at a DECREASING rate (concave)"))

print("\n2. BOTTOM TIME EFFECT:")
print(f"   Coefficient: {bt_coef:.4f} | Odds ratio: {np.exp(bt_coef):.4f}")
print(f"   Marginal effect: {bt_me:.4f} ({bt_me*100:.2f} pp per minute)")

print("\n3. DEPTH × BOTTOM TIME INTERACTION:")
print(f"   Coefficient: {interaction_coef:.6f}")
print(f"   Marginal effect: {interaction_me:.6f}")
print("   → " + ("SYNERGISTIC RISK: Longer time is more dangerous at greater depths"
              if interaction_coef > 0 else
              "Interaction suggests weaker incremental risk with time at greater depths"))

print("\n4. SAFETY STOP EFFECT:")
print(f"   Coefficient: {ss_coef:.4f} | Odds ratio: {np.exp(ss_coef):.4f}")
print(f"   Marginal effect: {ss_me:.4f} ({abs(ss_me)*100:.2f} pp reduction)")

print("\n5. ASCENT RATE EFFECT:")
print(f"   Coefficient: {ascent_coef:.4f} | Odds ratio: {np.exp(ascent_coef):.4f}")
print(f"   Marginal effect: {ascent_me:.4f} ({ascent_me*100:.2f} pp per m/min)")

print("\n6. NITROX EFFECT:")
print(f"   Coefficient: {nitrox_coef:.4f} | Odds ratio: {np.exp(nitrox_coef):.4f}")
print(f"   Marginal effect: {nitrox_me:.4f} ({abs(nitrox_me)*100:.2f} pp)")

print("\n7. NDL VIOLATION:")
print(f"   Coefficient: {ndl_coef:.4f} | Odds ratio: {np.exp(ndl_coef):.4f}")
print(f"   Marginal effect: {ndl_me:.4f} ({ndl_me*100:.2f} pp)")

# =========================
# Scenario Analysis
# =========================
print("\n" + "="*80)
print("SCENARIO ANALYSIS: PREDICTED INCIDENT PROBABILITIES")
print("="*80)

scenarios = [
    {'name': 'Safe Recreational Dive', 'MaxDepth_m': 18, 'BottomTime_min': 40,
     'AscentRate_m_per_min': 10, 'SafetyStop': 1, 'NDL_Violation': 0, 'Nitrox': 0},
    {'name': 'Deep Dive with Safety Stop', 'MaxDepth_m': 35, 'BottomTime_min': 25,
     'AscentRate_m_per_min': 10, 'SafetyStop': 1, 'NDL_Violation': 0, 'Nitrox': 1},
    {'name': 'Risky: Deep + Long + No Safety Stop', 'MaxDepth_m': 40, 'BottomTime_min': 35,
     'AscentRate_m_per_min': 12, 'SafetyStop': 0, 'NDL_Violation': 0, 'Nitrox': 0},
    {'name': 'Very Risky: Fast Ascent + NDL Violation', 'MaxDepth_m': 35, 'BottomTime_min': 30,
     'AscentRate_m_per_min': 20, 'SafetyStop': 0, 'NDL_Violation': 1, 'Nitrox': 0},
]

print("\nScenario predictions:")
for scenario in scenarios:
    sc_data = pd.DataFrame([scenario])
    sc_data['MaxDepth_m_sq'] = sc_data['MaxDepth_m'] ** 2
    sc_data['Depth_x_Time'] = sc_data['MaxDepth_m'] * sc_data['BottomTime_min']
    sc_data['I(MaxDepth_m ** 2)'] = sc_data['MaxDepth_m_sq']
    sc_data['MaxDepth_m:BottomTime_min'] = sc_data['Depth_x_Time']
    prob = float(logit_full.predict(sc_data)[0])
    print(f"\n  {scenario['name']}:")
    print(f"    Depth: {scenario['MaxDepth_m']}m, Time: {scenario['BottomTime_min']}min")
    print(f"    Ascent: {scenario['AscentRate_m_per_min']}m/min, Safety Stop: {'Yes' if scenario['SafetyStop'] else 'No'}")
    print(f"    NDL Violation: {'Yes' if scenario['NDL_Violation'] else 'No'}, Gas: {'Nitrox' if scenario['Nitrox'] else 'Air'}")
    print(f"    → Predicted incident probability: {prob:.4f} ({prob*100:.2f}%)")

# =========================
# HTML Report
# =========================
print("\n" + "="*80)
print("GENERATING COMPREHENSIVE HTML REPORT")
print("="*80)

html_report = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Comprehensive Diving Incident Analysis</title></head>
<body style="font-family:Segoe UI,Arial,sans-serif; background:#f5f5f5; margin:0; padding:30px;">
<div style="max-width:1200px; margin:0 auto; background:#fff; padding:30px; box-shadow:0 0 8px rgba(0,0,0,0.1);">
<h1>🤿 Comprehensive Diving Incident Analysis</h1>
<p><em>Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>

<h2>Executive Summary</h2>
<ul>
  <li><b>Total Dives:</b> {len(df):,}</li>
  <li><b>Incident Rate:</b> {df['Incident'].mean()*100:.1f}%</li>
  <li><b>Best ROC-AUC:</b> {perf_df['ROC-AUC'].max():.3f}</li>
</ul>

<h2>Dataset Overview</h2>
{summary_stats.to_html()}

<h3>Incident Rates by Key Factors</h3>
<ul>
  <li>With Safety Stop: {df[df['SafetyStop']==1]['Incident'].mean()*100:.1f}%</li>
  <li>Without Safety Stop: {df[df['SafetyStop']==0]['Incident'].mean()*100:.1f}%</li>
  <li>With NDL Violation: {df[df['NDL_Violation']==1]['Incident'].mean()*100:.1f}%</li>
  <li>Without NDL Violation: {df[df['NDL_Violation']==0]['Incident'].mean()*100:.1f}%</li>
  <li>Using Nitrox: {df[df['Nitrox']==1]['Incident'].mean()*100:.1f}%</li>
  <li>Using Air: {df[df['Nitrox']==0]['Incident'].mean()*100:.1f}%</li>
</ul>

<h2>Multicollinearity (VIF)</h2>
{vif_data.to_html(index=False)}

<h2>Model Comparison</h2>
{comparison.to_html(index=False)}

<h2>Full Logit Results</h2>
{logit_full.summary().as_html()}

<h2>Performance Metrics</h2>
{perf_df.to_html(index=False)}

<h2>Odds Ratios</h2>
{odds_ratios.to_html(index=False)}

<h2>Average Marginal Effects</h2>
{me_df.to_html(index=False)}

<h2>Visualizations</h2>
<img src="comprehensive_analysis_plots.png" style="max-width:100%; border:1px solid #ddd;">
<img src="partial_dependence_plots.png" style="max-width:100%; border:1px solid #ddd;">
<img src="residual_plots.png" style="max-width:100%; border:1px solid #ddd;">

<h2>Diagnostics</h2>
<p><b>Hosmer–Lemeshow:</b> χ² = {hl_chi2:.3f}, p = {hl_pval:.4f} — {'Good fit' if hl_pval > 0.05 else 'Poor fit'}</p>

</div></body></html>"""

with open("comprehensive_diving_analysis.html", "w", encoding="utf-8") as f:
    f.write(html_report)

print("✓ Comprehensive HTML report saved to: comprehensive_diving_analysis.html")

# =========================
# Save all outputs
# =========================
print("\n" + "="*80)
print("SAVING ALL OUTPUTS")
print("="*80)

comparison.to_csv("model_comparison.csv", index=False)
print("✓ Model comparison saved to: model_comparison.csv")

odds_ratios.to_csv("odds_ratios.csv", index=False)
print("✓ Odds ratios saved to: odds_ratios.csv")

me_df.to_csv("marginal_effects.csv", index=False)
print("✓ Marginal effects saved to: marginal_effects.csv")

perf_df.to_csv("model_performance.csv", index=False)
print("✓ Performance metrics saved to: model_performance.csv")

df[['DiveID', 'Incident', 'pred_prob_full', 'pred_class_full']].to_csv("predictions.csv", index=False)
print("✓ Predictions saved to: predictions.csv")

vif_data.to_csv("vif_scores.csv", index=False)
print("✓ VIF scores saved to: vif_scores.csv")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  1. comprehensive_diving_analysis.html  - Full HTML report with all results")
print("  2. comprehensive_analysis_plots.png    - ROC curves & probability curves")
print("  3. partial_dependence_plots.png        - Partial dependence for key variables")
print("  4. residual_plots.png                  - Residual diagnostics")
print("  5. model_comparison.csv                - Model fit statistics")
print("  6. odds_ratios.csv                     - Effect sizes with confidence intervals")
print("  7. marginal_effects.csv                - Average marginal effects (probability scale)")
print("  8. model_performance.csv               - Accuracy/Precision/Recall/F1/ROC-AUC")
print("  9. predictions.csv                     - Fitted probabilities & classes (Full Logit)")
print(" 10. vif_scores.csv                      - Multicollinearity check (VIF)")


