"""
Day 16: ARIMA Parameter Selection
==================================

Grid search for optimal ARIMA parameters using AIC/BIC information criteria
and automated parameter selection with pmdarima's auto_arima function.

Key Topics:
- Grid search across (p, d, q) parameter space
- AIC (Akaike Information Criterion) vs BIC (Bayesian Information Criterion)
- Automated parameter selection with auto_arima
- Model comparison and visualization
- Information criteria trade-offs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

print("="*70)
print("DAY 16: ARIMA PARAMETER SELECTION")
print("="*70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n" + "="*70)
print("1. LOADING AND PREPARING GOLD PRICE DATA")
print("="*70)

try:
    df = pd.read_csv('../day16/data/gold_prices.csv', parse_dates=['Date'])
except:
    try:
        df = pd.read_csv('./day16/data/gold_prices.csv', parse_dates=['Date'])
    except:
        print("⚠ Using fallback data loading")
        df = pd.read_csv('/home/tads/Work/TADSPROJ/30-days-of-tsa/day16/data/gold_prices.csv', 
                        parse_dates=['Date'])

# Clean data
if 'Price' not in df.columns:
    df = df.rename(columns={'Adj Close': 'Price'})
df = df.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)

print(f"✓ Data loaded: {len(df)} observations")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")

# Train-test split (80/20)
train_size = int(len(df) * 0.8)
train_data = df[:train_size].copy()
test_data = df[train_size:].copy()

print(f"\nTrain set: {len(train_data)} observations")
print(f"Test set: {len(test_data)} observations")
print(f"Train period: {train_data['Date'].min().date()} to {train_data['Date'].max().date()}")
print(f"Test period: {test_data['Date'].min().date()} to {test_data['Date'].max().date()}")

# ============================================================================
# 2. STATIONARITY CHECK
# ============================================================================
print("\n" + "="*70)
print("2. STATIONARITY ANALYSIS")
print("="*70)

def adf_test_verbose(series, name):
    """Perform ADF test with verbose output"""
    result = adfuller(series, autolag='AIC')
    is_stationary = result[1] <= 0.05
    
    print(f"\n{name}:")
    print(f"  ADF Statistic: {result[0]:.6f}")
    print(f"  P-value: {result[1]:.6f}")
    print(f"  Status: {'✓ STATIONARY' if is_stationary else '✗ NON-STATIONARY'}")
    
    return is_stationary, result[1]

# Test original series
prices = train_data['Price'].values
is_stat, p_val = adf_test_verbose(prices, "Original Series")

# Test first difference
diff1 = np.diff(prices, n=1)
is_stat_d1, p_val_d1 = adf_test_verbose(diff1, "First Difference (d=1)")

print(f"\n✓ Recommendation: d=1 (first differencing achieves stationarity)")

# ============================================================================
# 3. GRID SEARCH FOR OPTIMAL PARAMETERS
# ============================================================================
print("\n" + "="*70)
print("3. GRID SEARCH FOR OPTIMAL ARIMA PARAMETERS")
print("="*70)

# Define parameter ranges
p_range = range(0, 6)      # AR order: 0-5
d_range = range(0, 3)      # Differencing: 0-2
q_range = range(0, 6)      # MA order: 0-5

print(f"\nSearching parameter space:")
print(f"  p (AR): {list(p_range)}")
print(f"  d (differencing): {list(d_range)}")
print(f"  q (MA): {list(q_range)}")
print(f"  Total combinations: {len(p_range) * len(d_range) * len(q_range)}")

# Store results
grid_results = []
start_time = time.time()
successful_models = 0
failed_models = 0

print("\nFitting ARIMA models... (may take 1-2 minutes)")
print(f"{'Combo':<6} {'(p,d,q)':<12} {'AIC':<12} {'BIC':<12} {'Status':<20}")
print("-" * 65)

combo_count = 0
for p in p_range:
    for d in d_range:
        for q in q_range:
            combo_count += 1
            try:
                model = ARIMA(train_data['Price'], order=(p, d, q))
                fitted = model.fit()
                
                # Get AIC and BIC
                aic = fitted.aic
                bic = fitted.bic
                
                # Make forecast
                forecast = fitted.get_forecast(steps=len(test_data))
                forecast_values = forecast.predicted_mean.values
                
                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(test_data['Price'], forecast_values))
                
                grid_results.append({
                    'p': p,
                    'd': d,
                    'q': q,
                    'order': f"({p},{d},{q})",
                    'AIC': aic,
                    'BIC': bic,
                    'RMSE': rmse,
                    'Model': fitted
                })
                
                successful_models += 1
                status = "✓"
                
                if successful_models % 20 == 0:
                    print(f"{combo_count:<6} {f'({p},{d},{q})':<12} {aic:<12.2f} {bic:<12.2f} {status:<20}")
                    
            except Exception as e:
                failed_models += 1
                status = f"✗ Failed"

elapsed = time.time() - start_time

results_df = pd.DataFrame(grid_results)
print(f"\n✓ Completed in {elapsed:.1f} seconds")
print(f"✓ Successful models: {successful_models}")
print(f"✗ Failed models: {failed_models}")

# ============================================================================
# 4. INFORMATION CRITERIA ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("4. INFORMATION CRITERIA ANALYSIS")
print("="*70)

# Best by AIC
best_aic_idx = results_df['AIC'].idxmin()
best_aic = results_df.loc[best_aic_idx]

print(f"\nBest by AIC (favors fit quality):")
print(f"  Model: ARIMA{best_aic['order']}")
print(f"  AIC: {best_aic['AIC']:.2f}")
print(f"  BIC: {best_aic['BIC']:.2f}")
print(f"  RMSE: {best_aic['RMSE']:.2f}")

# Best by BIC
best_bic_idx = results_df['BIC'].idxmin()
best_bic = results_df.loc[best_bic_idx]

print(f"\nBest by BIC (favors parsimony):")
print(f"  Model: ARIMA{best_bic['order']}")
print(f"  AIC: {best_bic['AIC']:.2f}")
print(f"  BIC: {best_bic['BIC']:.2f}")
print(f"  RMSE: {best_bic['RMSE']:.2f}")

# Best by RMSE
best_rmse_idx = results_df['RMSE'].idxmin()
best_rmse = results_df.loc[best_rmse_idx]

print(f"\nBest by Test RMSE (best forecast):")
print(f"  Model: ARIMA{best_rmse['order']}")
print(f"  AIC: {best_rmse['AIC']:.2f}")
print(f"  BIC: {best_rmse['BIC']:.2f}")
print(f"  RMSE: {best_rmse['RMSE']:.2f}")

# Information Criteria Explanation
print(f"\nInformation Criteria Explanation:")
print(f"  AIC (Akaike): Balances fit vs complexity, penalizes model size")
print(f"  BIC (Bayesian): Stronger penalty for complexity, prefers simpler models")
print(f"  Formula: IC = -2*log(L) + k*penalty")
print(f"    where L = likelihood, k = parameters")
print(f"    AIC penalty: 2k")
print(f"    BIC penalty: k*log(n)")
print(f"  Trade-off: AIC may overfit, BIC may underfit")

# ============================================================================
# 5. TOP MODELS COMPARISON
# ============================================================================
print("\n" + "="*70)
print("5. TOP 10 MODELS BY BIC")
print("="*70)

top_models = results_df.nsmallest(10, 'BIC')[['order', 'AIC', 'BIC', 'RMSE']]
print("\n", top_models.to_string(index=False))

# ============================================================================
# 6. AUTO_ARIMA AUTOMATED SELECTION
# ============================================================================
print("\n" + "="*70)
print("6. AUTOMATED PARAMETER SELECTION WITH AUTO_ARIMA")
print("="*70)

print("\nRunning auto_arima (stepwise optimization)...")
auto_model = auto_arima(
    train_data['Price'],
    start_p=0, max_p=5,
    start_d=0, max_d=2,
    start_q=0, max_q=5,
    seasonal=False,
    stepwise=True,
    information_criterion='bic',
    trace=False,
    error_action='ignore',
    maxiter=50
)

auto_order = auto_model.order
auto_aic = auto_model.aic()
auto_bic = auto_model.bic()

print(f"\n✓ Auto ARIMA Result:")
print(f"  Model: ARIMA{auto_order}")
print(f"  AIC: {auto_aic:.2f}")
print(f"  BIC: {auto_bic:.2f}")

# Forecast with auto_arima
auto_forecast = auto_model.predict(n_periods=len(test_data))
auto_rmse = np.sqrt(mean_squared_error(test_data['Price'], auto_forecast))
auto_mae = mean_absolute_error(test_data['Price'], auto_forecast)

print(f"  Test RMSE: {auto_rmse:.2f}")
print(f"  Test MAE: {auto_mae:.2f}")

# ============================================================================
# 7. GRID SEARCH VS AUTO_ARIMA COMPARISON
# ============================================================================
print("\n" + "="*70)
print("7. GRID SEARCH VS AUTO_ARIMA COMPARISON")
print("="*70)

comparison_data = {
    'Method': ['Grid Search (AIC)', 'Grid Search (BIC)', 'Grid Search (RMSE)', 'Auto ARIMA'],
    'Model': [
        f"ARIMA{best_aic['order']}",
        f"ARIMA{best_bic['order']}",
        f"ARIMA{best_rmse['order']}",
        f"ARIMA{auto_order}"
    ],
    'AIC': [
        f"{best_aic['AIC']:.2f}",
        f"{best_bic['AIC']:.2f}",
        f"{best_rmse['AIC']:.2f}",
        f"{auto_aic:.2f}"
    ],
    'BIC': [
        f"{best_aic['BIC']:.2f}",
        f"{best_bic['BIC']:.2f}",
        f"{best_rmse['BIC']:.2f}",
        f"{auto_bic:.2f}"
    ],
    'Test RMSE': [
        f"{best_aic['RMSE']:.2f}",
        f"{best_bic['RMSE']:.2f}",
        f"{best_rmse['RMSE']:.2f}",
        f"{auto_rmse:.2f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n", comparison_df.to_string(index=False))

print(f"\nKey Observations:")
print(f"  • AIC selects more complex models (better fit)")
print(f"  • BIC selects simpler models (avoid overfitting)")
print(f"  • Auto ARIMA uses stepwise selection (fast, good balance)")
print(f"  • RMSE may not correlate with AIC/BIC")
print(f"  • Different criteria can select different optimal models")

# ============================================================================
# 8. VISUALIZATION: AIC/BIC HEATMAP
# ============================================================================
print("\n" + "="*70)
print("8. VISUALIZATION: PARAMETER SELECTION LANDSCAPE")
print("="*70)

# Create heatmap for d=1 (most common)
d_target = 1
heatmap_data = results_df[results_df['d'] == d_target].copy()

# Pivot for heatmap
aic_pivot = heatmap_data.pivot_table(values='AIC', index='p', columns='q')
bic_pivot = heatmap_data.pivot_table(values='BIC', index='p', columns='q')

fig = sp.make_subplots(
    rows=1, cols=2,
    subplot_titles=(f'AIC Values (d={d_target})', f'BIC Values (d={d_target})'),
    specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}]]
)

# AIC heatmap
fig.add_trace(
    go.Heatmap(
        z=aic_pivot.values,
        x=aic_pivot.columns,
        y=aic_pivot.index,
        colorscale='Viridis',
        name='AIC',
        colorbar=dict(x=0.46, len=0.4)
    ),
    row=1, col=1
)

# BIC heatmap
fig.add_trace(
    go.Heatmap(
        z=bic_pivot.values,
        x=bic_pivot.columns,
        y=bic_pivot.index,
        colorscale='Viridis',
        name='BIC',
        colorbar=dict(x=1.02, len=0.4)
    ),
    row=1, col=2
)

fig.update_xaxes(title_text="MA Order (q)", row=1, col=1)
fig.update_yaxes(title_text="AR Order (p)", row=1, col=1)
fig.update_xaxes(title_text="MA Order (q)", row=1, col=2)
fig.update_yaxes(title_text="AR Order (p)", row=1, col=2)

fig.update_layout(height=500, width=1000, showlegend=False)
fig.write_html("aic_bic_heatmap.html")
print("\n✓ Saved: aic_bic_heatmap.html")

# ============================================================================
# 9. VISUALIZATION: AIC vs BIC SCATTER
# ============================================================================

fig = go.Figure()

# Scatter plot: AIC vs BIC
fig.add_trace(go.Scatter(
    x=results_df['AIC'],
    y=results_df['BIC'],
    mode='markers',
    marker=dict(
        size=6,
        color=results_df['RMSE'],
        colorscale='Plasma',
        showscale=True,
        colorbar=dict(title="Test RMSE")
    ),
    text=results_df['order'],
    hovertemplate='<b>%{text}</b><br>AIC: %{x:.2f}<br>BIC: %{y:.2f}<extra></extra>'
))

# Highlight best models
fig.add_trace(go.Scatter(
    x=[best_aic['AIC']],
    y=[best_aic['BIC']],
    mode='markers+text',
    marker=dict(size=15, color='red', symbol='star'),
    text=['Best AIC'],
    textposition='top center',
    name='Best AIC',
    showlegend=True
))

fig.add_trace(go.Scatter(
    x=[best_bic['AIC']],
    y=[best_bic['BIC']],
    mode='markers+text',
    marker=dict(size=15, color='green', symbol='star'),
    text=['Best BIC'],
    textposition='top center',
    name='Best BIC',
    showlegend=True
))

fig.add_trace(go.Scatter(
    x=[auto_aic],
    y=[auto_bic],
    mode='markers+text',
    marker=dict(size=15, color='blue', symbol='diamond'),
    text=['Auto ARIMA'],
    textposition='top center',
    name='Auto ARIMA',
    showlegend=True
))

fig.update_layout(
    title="AIC vs BIC: Information Criteria Trade-off",
    xaxis_title="AIC (lower is better, favors fit)",
    yaxis_title="BIC (lower is better, favors parsimony)",
    height=600,
    hovermode='closest'
)
fig.write_html("aic_bic_scatter.html")
print("✓ Saved: aic_bic_scatter.html")

# ============================================================================
# 10. SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("9. SUMMARY AND RECOMMENDATIONS")
print("="*70)

print(f"\n📊 Grid Search Summary:")
print(f"  • Parameter combinations tested: {len(results_df)}")
print(f"  • Best model (BIC): ARIMA{best_bic['order']}")
print(f"  • Best model (RMSE): ARIMA{best_rmse['order']}")

print(f"\n🤖 Auto ARIMA Summary:")
print(f"  • Selected model: ARIMA{auto_order}")
print(f"  • Computation time: ~1-2 minutes (much faster than grid search)")
print(f"  • Uses stepwise algorithm for efficiency")

print(f"\n💡 When to Use Each Method:")
print(f"  Grid Search:")
print(f"    ✓ Small parameter spaces (p,d,q ≤ 5)")
print(f"    ✓ When you want to explore all options")
print(f"    ✓ Educational purposes")
print(f"    ✓ Understanding parameter landscape")
print(f"    ✗ Computationally expensive for large spaces")
print(f"\n  Auto ARIMA:")
print(f"    ✓ Fast, automatic selection")
print(f"    ✓ Practical for production systems")
print(f"    ✓ Large parameter spaces")
print(f"    ✓ No manual tuning required")
print(f"    ✗ Less interpretable (black box)")

print(f"\n📈 Information Criteria Recommendation:")
print(f"  • Use AIC when: Prioritizing predictive accuracy")
print(f"  • Use BIC when: Prioritizing interpretability and avoiding overfitting")
print(f"  • Both agree on model?: High confidence in selection")
print(f"  • They disagree?: Consider both models, validate on holdout set")

print(f"\n✓ Day 16 Analysis Complete!")
print("="*70)
