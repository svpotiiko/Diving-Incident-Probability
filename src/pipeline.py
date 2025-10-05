"""Main pipeline orchestrating the end-to-end analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from src.io_utils import (load_config, load_data, ensure_output_dir, 
                          save_dataframe, save_figure, save_html_report)
from src.features import engineer_features, get_feature_names, prepare_prediction_data
from src.modeling import (fit_baseline_logit, fit_full_logit, fit_full_probit,
                          fit_robust_logit, calculate_marginal_effects,
                          calculate_odds_ratios, calculate_vif, 
                          create_model_comparison, likelihood_ratio_test)
from src.metrics import (add_predictions, calculate_performance_metrics,
                        calculate_confusion_matrices, hosmer_lemeshow_test,
                        perform_cross_validation, get_roc_data)
from src.viz import (setup_plotting_style, create_main_plots, 
                     create_partial_dependence_plots, create_residual_plots)
from src.report import generate_html_report, print_summary_report


def run_analysis(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Run complete diving risk analysis pipeline.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Dictionary containing all analysis results
    """
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Setup output directory
    ensure_output_dir(config)
    
    # Setup plotting
    setup_plotting_style(config)
    
    # Load data
    print("Loading data...")
    df = load_data(config)
    
    print(f"Loaded {len(df)} dives with incident rate: {df['Incident'].mean():.2%}")
    
    # Feature engineering
    print("Engineering features...")
    df = engineer_features(df, config)
    feature_names = get_feature_names(config)
    
    # Summary statistics
    summary_cols = ['MaxDepth_m', 'BottomTime_min', 'AscentRate_m_per_min', 
                    'SafetyStop', 'NDL_Violation', 'Nitrox', 'Incident']
    summary_stats = df[summary_cols].describe()
    
    # Multicollinearity check
    print("Checking multicollinearity (VIF)...")
    vif_data = calculate_vif(df, feature_names)
    
    # Fit models
    print("Fitting models...")
    print("  - Baseline Logit")
    logit_baseline = fit_baseline_logit(df, config)
    
    print("  - Full Logit")
    logit_full = fit_full_logit(df, config)
    
    print("  - Full Probit")
    probit_full = fit_full_probit(df, config)
    
    print("  - Robust Logit")
    logit_robust = fit_robust_logit(df, config)
    
    models = {
        'Baseline Logit': logit_baseline,
        'Full Logit': logit_full,
        'Full Probit': probit_full,
        'Full Logit (Robust SE)': logit_robust
    }
    
    # Model comparison
    print("Comparing models...")
    comparison = create_model_comparison(models)
    
    # Likelihood ratio test
    lr_stat, lr_df, lr_pval = likelihood_ratio_test(logit_baseline, logit_full)
    lr_test_results = {
        'lr_stat': lr_stat,
        'df': lr_df,
        'pval': lr_pval
    }
    
    # Marginal effects
    print("Calculating marginal effects...")
    me_df = calculate_marginal_effects(logit_full)
    
    # Odds ratios
    print("Calculating odds ratios...")
    odds_ratios = calculate_odds_ratios(logit_full)
    
    # Add predictions
    print("Generating predictions...")
    threshold = config['analysis']['classification_threshold']
    
    # Add predictions to dataframe for metrics
    df['pred_prob_baseline'] = logit_baseline.predict()
    df['pred_prob_full'] = logit_full.predict()
    df['pred_prob_probit'] = probit_full.predict()
    
    # Performance metrics
    print("Calculating performance metrics...")
    models_for_metrics = {
        'Baseline Logit': logit_baseline,
        'Full Logit': logit_full,
        'Full Probit': probit_full
    }
    perf_df = calculate_performance_metrics(df, models_for_metrics, threshold)
    
    # Confusion matrices
    print("Computing confusion matrices...")
    confusion_matrices = calculate_confusion_matrices(df, models_for_metrics, threshold)
    
    # ROC data
    print("Computing ROC curves...")
    roc_data = get_roc_data(df, models_for_metrics)
    
    # Hosmer-Lemeshow test
    print("Performing Hosmer-Lemeshow test...")
    hl_groups = config['analysis']['hosmer_lemeshow_groups']
    hl_chi2, hl_df, hl_pval = hosmer_lemeshow_test(
        df['Incident'].values, 
        df['pred_prob_full'].values, 
        g=hl_groups
    )
    hl_results = {
        'chi2': hl_chi2,
        'df': hl_df,
        'pval': hl_pval
    }
    
    # Cross-validation
    print("Performing cross-validation...")
    cv_results = perform_cross_validation(df, feature_names, config)
    
    # Create visualizations
    print("Creating visualizations...")
    
    print("  - Main analysis plots")
    fig_main = create_main_plots(df, models_for_metrics, roc_data, 
                                  confusion_matrices, config)
    save_figure(fig_main, config, 'main_plots')
    plt.close(fig_main)
    
    print("  - Partial dependence plots")
    fig_partial = create_partial_dependence_plots(df, logit_full, config)
    save_figure(fig_partial, config, 'partial_plots')
    plt.close(fig_partial)
    
    print("  - Residual plots")
    fig_residual = create_residual_plots(df, logit_full)
    save_figure(fig_residual, config, 'residual_plots')
    plt.close(fig_residual)
    
    # Generate HTML report
    print("Generating HTML report...")
    html_content = generate_html_report(
        df, summary_stats, vif_data, comparison, logit_full,
        perf_df, odds_ratios, me_df, hl_chi2, hl_pval, config
    )
    save_html_report(html_content, config)
    
    # Save CSV outputs
    print("Saving CSV outputs...")
    save_dataframe(comparison, config, 'model_comparison')
    save_dataframe(odds_ratios, config, 'odds_ratios')
    save_dataframe(me_df, config, 'marginal_effects')
    save_dataframe(perf_df, config, 'performance')
    save_dataframe(vif_data, config, 'vif_scores')
    
    # Save predictions
    pred_df = df[['DiveID', 'Incident', 'pred_prob_full']].copy()
    pred_df['pred_class_full'] = (pred_df['pred_prob_full'] > threshold).astype(int)
    save_dataframe(pred_df, config, 'predictions')
    
    # Print summary to console
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - PRINTING SUMMARY")
    print("="*80)
    
    print_summary_report(
        df, models, perf_df, me_df, vif_data, comparison,
        confusion_matrices, hl_results, cv_results, 
        lr_test_results, config
    )
    
    # Scenario analysis
    print("\n" + "="*80)
    print("SCENARIO ANALYSIS")
    print("="*80)
    
    for scenario in config['scenarios']:
        scenario_data = prepare_prediction_data(scenario)
        prob = float(logit_full.predict(scenario_data)[0])
        
        print(f"\n{scenario['name']}:")
        print(f"  Depth: {scenario['MaxDepth_m']}m, Time: {scenario['BottomTime_min']}min")
        print(f"  Safety Stop: {'Yes' if scenario['SafetyStop'] else 'No'}, "
              f"NDL Violation: {'Yes' if scenario['NDL_Violation'] else 'No'}")
        print(f"  → Predicted incident probability: {prob:.4f} ({prob*100:.2f}%)")
    
    # Print file locations
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files in outputs/:")
    for key, filename in config['output_files'].items():
        print(f"  - {filename}")
    
    # Return results dictionary
    results = {
        'df': df,
        'config': config,
        'models': models,
        'summary_stats': summary_stats,
        'vif_data': vif_data,
        'comparison': comparison,
        'marginal_effects': me_df,
        'odds_ratios': odds_ratios,
        'performance': perf_df,
        'confusion_matrices': confusion_matrices,
        'roc_data': roc_data,
        'hosmer_lemeshow': hl_results,
        'cross_validation': cv_results,
        'likelihood_ratio_test': lr_test_results
    }
    
    return results