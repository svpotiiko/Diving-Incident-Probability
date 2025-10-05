"""HTML report generation for diving risk analysis."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any


def generate_html_report(df: pd.DataFrame, summary_stats: pd.DataFrame,
                        vif_data: pd.DataFrame, comparison: pd.DataFrame,
                        logit_full, perf_df: pd.DataFrame, 
                        odds_ratios: pd.DataFrame, me_df: pd.DataFrame,
                        hl_chi2: float, hl_pval: float,
                        config: Dict[str, Any]) -> str:
    """Generate comprehensive HTML report.
    
    Args:
        df: Main DataFrame with data
        summary_stats: Summary statistics DataFrame
        vif_data: VIF scores DataFrame
        comparison: Model comparison DataFrame
        logit_full: Full logit model object
        perf_df: Performance metrics DataFrame
        odds_ratios: Odds ratios DataFrame
        me_df: Marginal effects DataFrame
        hl_chi2: Hosmer-Lemeshow chi-square statistic
        hl_pval: Hosmer-Lemeshow p-value
        config: Configuration dictionary
        
    Returns:
        HTML string
    """
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Comprehensive Diving Incident Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #f5f5f5;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #fff;
            padding: 30px;
            box-shadow: 0 0 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 10px 15px;
            background: #ecf0f1;
            border-radius: 5px;
        }}
        .metric strong {{
            color: #2c3e50;
        }}
        img {{
            max-width: 100%;
            border: 1px solid #ddd;
            margin: 20px 0;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
<div class="container">

<h1>🤿 Comprehensive Diving Incident Analysis</h1>
<p><em>Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>

<h2>Executive Summary</h2>
<div class="metric"><strong>Total Dives:</strong> {len(df):,}</div>
<div class="metric"><strong>Incident Rate:</strong> {df['Incident'].mean()*100:.1f}%</div>
<div class="metric"><strong>Best ROC-AUC:</strong> {perf_df['ROC-AUC'].max():.3f}</div>

<h2>Dataset Overview</h2>
{summary_stats.to_html()}

<h3>Incident Rates by Key Factors</h3>
<ul>
    <li><strong>With Safety Stop:</strong> {df[df['SafetyStop']==1]['Incident'].mean()*100:.1f}%</li>
    <li><strong>Without Safety Stop:</strong> {df[df['SafetyStop']==0]['Incident'].mean()*100:.1f}%</li>
    <li><strong>With NDL Violation:</strong> {df[df['NDL_Violation']==1]['Incident'].mean()*100:.1f}%</li>
    <li><strong>Without NDL Violation:</strong> {df[df['NDL_Violation']==0]['Incident'].mean()*100:.1f}%</li>
    <li><strong>Using Nitrox:</strong> {df[df['Nitrox']==1]['Incident'].mean()*100:.1f}%</li>
    <li><strong>Using Air:</strong> {df[df['Nitrox']==0]['Incident'].mean()*100:.1f}%</li>
</ul>

<h2>Multicollinearity Check (VIF)</h2>
{vif_data.to_html(index=False)}
<p><em>Note: VIF &gt; 10 suggests problematic multicollinearity. Max VIF: {vif_data['VIF'].max():.2f}</em></p>

<h2>Model Comparison</h2>
{comparison.to_html(index=False)}

<h2>Full Logit Model Results</h2>
{logit_full.summary().as_html()}

<h2>Performance Metrics</h2>
{perf_df.to_html(index=False)}

<h2>Odds Ratios (Full Logit Model)</h2>
{odds_ratios.to_html(index=False)}
<p><em>Odds ratio &gt; 1 indicates increased risk; &lt; 1 indicates decreased risk</em></p>

<h2>Average Marginal Effects</h2>
{me_df.to_html(index=False)}
<p><em>Marginal effects represent the change in probability (in percentage points) for a one-unit increase in the predictor</em></p>

<h2>Goodness-of-Fit: Hosmer-Lemeshow Test</h2>
<p><strong>Chi-square statistic:</strong> {hl_chi2:.4f}</p>
<p><strong>P-value:</strong> {hl_pval:.4f}</p>
<p><strong>Result:</strong> {'Good fit (p > 0.05)' if hl_pval > 0.05 else 'Poor fit (p ≤ 0.05)'}</p>

<h2>Visualizations</h2>

<h3>Main Analysis Plots</h3>
<img src="{config['output_files']['main_plots']}" alt="Main Analysis Plots">

<h3>Partial Dependence Plots</h3>
<img src="{config['output_files']['partial_plots']}" alt="Partial Dependence Plots">

<h3>Residual Diagnostics</h3>
<img src="{config['output_files']['residual_plots']}" alt="Residual Plots">

<div class="footer">
    <p>Report generated by Dive Risk Analysis Pipeline</p>
    <p>Configuration: {config['paths']['input_csv']}</p>
</div>

</div>
</body>
</html>"""
    
    return html


def print_summary_report(df: pd.DataFrame, models: Dict[str, Any],
                        perf_df: pd.DataFrame, me_df: pd.DataFrame,
                        vif_data: pd.DataFrame, comparison: pd.DataFrame,
                        confusion_matrices: Dict[str, Dict],
                        hl_results: Dict[str, float],
                        cv_results: Dict[str, Any],
                        lr_test_results: Dict[str, Any],
                        config: Dict[str, Any]) -> None:
    """Print comprehensive summary report to console.
    
    Args:
        df: Main DataFrame
        models: Dictionary of fitted models
        perf_df: Performance metrics
        me_df: Marginal effects
        vif_data: VIF scores
        comparison: Model comparison
        confusion_matrices: Confusion matrix data
        hl_results: Hosmer-Lemeshow test results
        cv_results: Cross-validation results
        lr_test_results: Likelihood ratio test results
        config: Configuration dictionary
    """
    print("="*80)
    print("COMPREHENSIVE DIVING INCIDENT ANALYSIS")
    print("="*80)
    print(f"\nDataset: {len(df)} dives")
    print(f"Incident rate: {df['Incident'].mean():.2%}")
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    summary_cols = ['MaxDepth_m', 'BottomTime_min', 'AscentRate_m_per_min', 
                    'SafetyStop', 'NDL_Violation', 'Nitrox', 'Incident']
    print(df[summary_cols].describe())
    
    print("\nIncident rates by key factors:")
    print(f"  With Safety Stop: {df[df['SafetyStop']==1]['Incident'].mean():.2%}")
    print(f"  Without Safety Stop: {df[df['SafetyStop']==0]['Incident'].mean():.2%}")
    print(f"  With NDL Violation: {df[df['NDL_Violation']==1]['Incident'].mean():.2%}")
    print(f"  Without NDL Violation: {df[df['NDL_Violation']==0]['Incident'].mean():.2%}")
    
    print("\n" + "="*80)
    print("MULTICOLLINEARITY CHECK (VIF)")
    print("="*80)
    print(vif_data.to_string(index=False))
    print(f"\nMax VIF: {vif_data['VIF'].max():.2f}")
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(comparison.to_string(index=False))
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE METRICS")
    print("="*80)
    print(perf_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("CONFUSION MATRICES")
    print("="*80)
    for model_name, cm_data in confusion_matrices.items():
        cm = cm_data['confusion_matrix']
        print(f"\n{model_name}:")
        print(f"                 Predicted No  Predicted Yes")
        print(f"  Actual No      {cm[0,0]:8d}      {cm[0,1]:8d}")
        print(f"  Actual Yes     {cm[1,0]:8d}      {cm[1,1]:8d}")
        print(f"  Sensitivity: {cm_data['sensitivity']:.3f}")
        print(f"  Specificity: {cm_data['specificity']:.3f}")
    
    print("\n" + "="*80)
    print("HOSMER-LEMESHOW GOODNESS-OF-FIT TEST")
    print("="*80)
    print(f"Chi-square: {hl_results['chi2']:.4f}")
    print(f"P-value: {hl_results['pval']:.4f}")
    print(f"Result: {'Good fit' if hl_results['pval'] > 0.05 else 'Poor fit'}")
    
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS")
    print("="*80)
    print(f"ROC-AUC scores: {cv_results['scores']}")
    print(f"Mean: {cv_results['mean']:.4f} (+/- {cv_results['std'] * 2:.4f})")
    
    print("\n" + "="*80)
    print("LIKELIHOOD RATIO TEST")
    print("="*80)
    print(f"LR statistic: {lr_test_results['lr_stat']:.4f}")
    print(f"Degrees of freedom: {lr_test_results['df']}")
    print(f"P-value: {lr_test_results['pval']:.6f}")
    print(f"Result: {'Full model is better' if lr_test_results['pval'] < 0.05 else 'Models similar'}")
    
    print("\n" + "="*80)
    print("AVERAGE MARGINAL EFFECTS")
    print("="*80)
    print(me_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    _print_key_findings(models['Full Logit'], me_df)


def _print_key_findings(logit_full, me_df: pd.DataFrame) -> None:
    """Print interpretation of key findings."""
    params = logit_full.params
    
    print("\n1. DEPTH EFFECT:")
    depth_coef = params['MaxDepth_m']
    depth_sq_coef = params['I(MaxDepth_m ** 2)']
    depth_me = me_df.loc[me_df['Variable'] == 'MaxDepth_m', 'Marginal Effect'].values[0]
    print(f"   Linear coef: {depth_coef:.4f}, Quadratic: {depth_sq_coef:.6f}")
    print(f"   Marginal effect: {depth_me:.4f} ({depth_me*100:.2f} pp per meter)")
    print("   → " + ("Risk ACCELERATES at greater depths" if depth_sq_coef > 0 
                     else "Risk increases but at decreasing rate"))
    
    print("\n2. SAFETY STOP EFFECT:")
    ss_coef = params['SafetyStop']
    ss_me = me_df.loc[me_df['Variable'] == 'SafetyStop', 'Marginal Effect'].values[0]
    print(f"   Coefficient: {ss_coef:.4f}, OR: {np.exp(ss_coef):.4f}")
    print(f"   Marginal effect: {ss_me:.4f} ({abs(ss_me)*100:.2f} pp reduction)")
    
    print("\n3. NDL VIOLATION:")
    ndl_coef = params['NDL_Violation']
    ndl_me = me_df.loc[me_df['Variable'] == 'NDL_Violation', 'Marginal Effect'].values[0]
    print(f"   Coefficient: {ndl_coef:.4f}, OR: {np.exp(ndl_coef):.4f}")
    print(f"   Marginal effect: {ndl_me:.4f} ({ndl_me*100:.2f} pp increase)")
    
    print("\n4. DEPTH × BOTTOM TIME INTERACTION:")
    int_coef = params['MaxDepth_m:BottomTime_min']
    print(f"   Coefficient: {int_coef:.6f}")
    print("   → " + ("SYNERGISTIC RISK: Longer time is more dangerous at greater depths"
                     if int_coef > 0 else "Interaction suggests weaker effect"))