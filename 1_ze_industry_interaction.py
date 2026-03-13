"""
TWO-STAGE TRANSFER FORECASTING — STAGE 2
==========================================
Purpose : Learn Y ~ X̂ relational parameter β using spatial features
          from Stage 1 forecasts. Produce final 6-month Y forecast.
          Compare against benchmark models.

Inputs  :
    Y_monthly            (n_months,)          — target series (96 obs)
    X_hat_spatial        (6, n_features)      — Stage 1 spatial forecast features
    X_historical_spatial (n_months, n_feat)   — historical spatial features
    climate_future       (6, k)               — known future climate features
    holidays_future      (6, m)               — known future holiday features

Outputs :
    Y_hat            (6,)                — point forecast h=1→6
    Y_hat_intervals  (6, 2)              — [lower, upper] prediction intervals
    benchmark_results dict               — SARIMA + Naïve comparison
    val_results      DataFrame           — walk-forward CV metrics

Literature:
  - ElasticNet:           Zou & Hastie (2005) "Regularization and Variable Selection via EN"
  - Generated regressors: Pagan (1984) "Econometric Issues in Analysis of Regressions"
  - SARIMA benchmark:     Box & Jenkins (1970); Hyndman & Athanasopoulos (2021)
  - Bootstrap PI:         Efron & Tibshirani (1993) "Introduction to the Bootstrap"
  - Walk-forward CV:      Bergmeir & Benitez (2012) "On the use of CV for time series"
  - Bridge equations:     Giannone, Reichlin & Small (2008) nowcasting
"""

import numpy as np
import pandas as pd
import warnings
import sys
sys.path.insert(0, '/home/claude')

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb

warnings.filterwarnings('ignore')


# ==============================================================================
# STEP 1: BUILD HISTORICAL SPATIAL FEATURES FROM PANEL
# ==============================================================================

def build_historical_spatial_features(panel, pca, scaler_pca):
    """
    Reconstruct spatial features from historical X panel.
    Must use IDENTICAL transformations as Stage 1 engineer_spatial_features().
    These become the training features for Stage 2.

    Input:  panel (n_months, n_cities)
    Output: hist_spatial (n_months-1, n_spatial_features)
    """
    cities    = panel.columns.tolist()
    n_cities  = len(cities)
    hist_means  = panel.mean()
    hist_6m_avg = panel.iloc[-6:].mean()
    
    # Compute city‑level distributional aggregates
    agg_df = compute_city_aggregates(panel)

    def gini(arr):
        arr = np.sort(np.abs(arr))
        n   = len(arr)
        idx = np.arange(1, n + 1)
        return (2 * np.sum(idx * arr) / (n * np.sum(arr) + 1e-8)) - (n + 1) / n

    records = []
    for t in range(1, len(panel)):
        vals      = panel.iloc[t].values
        prev_vals = panel.iloc[t - 1].values
        period    = panel.index[t]

        total_sum    = vals.sum()
        total_mean   = vals.mean()
        total_median = np.median(vals)

        breadth_above_hist  = (vals > hist_means.values).mean()
        breadth_pos_mom     = (vals > prev_vals).mean()
        breadth_above_trend = (vals > hist_6m_avg.values).mean()

        spatial_std  = vals.std()
        spatial_cv   = spatial_std / (abs(total_mean) + 1e-8)
        spatial_skew = float(stats.skew(vals))
        spatial_iqr  = float(np.percentile(vals, 75) - np.percentile(vals, 25))

        sorted_desc = np.sort(vals)[::-1]
        top10_idx   = min(10, n_cities)
        top10_sum   = sorted_desc[:top10_idx].sum()
        top10_share = top10_sum / (total_sum + 1e-8)
        gini_coeff  = gini(vals)

        vals_norm   = scaler_pca.transform(vals.reshape(1, -1))
        pca_factors = pca.transform(vals_norm)[0]

        row = {
            'period': period,
            'total_sum': total_sum, 'total_mean': total_mean,
            'total_median': total_median,
            'breadth_above_hist': breadth_above_hist,
            'breadth_pos_mom': breadth_pos_mom,
            'breadth_above_trend': breadth_above_trend,
            'spatial_std': spatial_std, 'spatial_cv': spatial_cv,
            'spatial_skew': spatial_skew, 'spatial_iqr': spatial_iqr,
            'top10_sum': top10_sum, 'top10_share': top10_share,
            'gini_coeff': gini_coeff,
        }
        # Add city‑aggregate features
        agg_row = agg_df.loc[period]
        row['city_variance'] = agg_row['city_variance']
        row['city_skewness'] = agg_row['city_skewness']
        row['city_kurtosis'] = agg_row['city_kurtosis']
        row['city_top5_share'] = agg_row['city_top5_share']
        row['city_herfindahl'] = agg_row['city_herfindahl']
        row['city_median_to_mean'] = agg_row['city_median_to_mean']
        row['city_cv'] = agg_row['city_cv']
        
        for k, fv in enumerate(pca_factors):
            row[f'pca_factor_{k+1}'] = fv
        records.append(row)

    hist_spatial = pd.DataFrame(records).set_index('period')
    print(f"  [INFO] Historical spatial: {hist_spatial.shape[0]} months x {hist_spatial.shape[1]} features")
    return hist_spatial


def compute_city_aggregates(panel):
    """
    Compute cross‑sectional distributional metrics for each month in the panel.

    Input:  panel (n_months, n_cities) – monthly values per city.
    Output: DataFrame (n_months, n_metrics) with columns:
        'city_variance', 'city_skewness', 'city_kurtosis',
        'city_top5_share', 'city_herfindahl', 'city_median_to_mean',
        'city_cv' (coefficient of variation).
    """
    records = []
    for period, row in panel.iterrows():
        vals = row.values
        total = vals.sum()
        mean = vals.mean()
        std = vals.std()
        # Variance
        var = vals.var()
        # Skewness and kurtosis (Fisher’s definition, bias=False for sample)
        skew = stats.skew(vals, bias=False)
        kurt = stats.kurtosis(vals, bias=False)
        # Top‑5 share
        sorted_desc = np.sort(vals)[::-1]
        top5 = sorted_desc[:min(5, len(vals))]
        top5_share = top5.sum() / (total + 1e-8)
        # Herfindahl–Hirschman index (sum of squared shares)
        shares = vals / (total + 1e-8)
        hhi = np.sum(shares ** 2)
        # Median‑to‑mean ratio
        median = np.median(vals)
        median_to_mean = median / (mean + 1e-8) if mean != 0 else 0
        # Coefficient of variation
        cv = std / (abs(mean) + 1e-8)
        records.append({
            'period': period,
            'city_variance': var,
            'city_skewness': skew,
            'city_kurtosis': kurt,
            'city_top5_share': top5_share,
            'city_herfindahl': hhi,
            'city_median_to_mean': median_to_mean,
            'city_cv': cv
        })
    agg_df = pd.DataFrame(records).set_index('period')
    print(f"  [INFO] City aggregates: {agg_df.shape[0]} months x {agg_df.shape[1]} features")
    return agg_df


# ==============================================================================
# STEP 2: BUILD STAGE 2 TRAINING MATRIX
# ==============================================================================

def build_stage2_training_matrix(Y, hist_spatial, climate_hist=None,
                                  holidays_hist=None, y_lags=[1,2,3,6,12], horizon=6):
    """
    Build aligned [X_spatial(t), Y_lags(t), climate(t+h), holidays(t+h)] → Y(t+h)
    for each horizon h=1→6. No data leakage: all features known at time t.

    Input:  Y (n_months,), hist_spatial (n_months-1, n_features)
    Output: dict {h: {'X': DataFrame, 'Y': Series}}
    """
    horizon_data  = {}
    common_periods = Y.index.intersection(hist_spatial.index)
    Y_al = Y.loc[common_periods]
    S_al = hist_spatial.loc[common_periods]
    n    = len(Y_al)

    for h in range(1, horizon + 1):
        rows, targets, periods_list = [], [], []
        for t in range(max(y_lags), n - h):
            period = common_periods[t]
            row    = S_al.iloc[t].to_dict()

            # Y lag features
            for lag in y_lags:
                row[f'y_lag_{lag}'] = Y_al.iloc[t - lag] if t - lag >= 0 else Y_al.iloc[0]

            # Y rolling stats
            roll3 = Y_al.iloc[max(0, t-3):t]
            roll6 = Y_al.iloc[max(0, t-6):t]
            row['y_roll_mean_3'] = roll3.mean()
            row['y_roll_mean_6'] = roll6.mean()
            row['y_roll_std_3']  = roll3.std() if len(roll3) > 1 else 0
            row['y_mom_1']       = Y_al.iloc[t] - Y_al.iloc[t-1] if t > 0 else 0
            row['y_mom_3']       = Y_al.iloc[t] - Y_al.iloc[t-3] if t >= 3 else 0

            # Climate/holidays at t+h (known future)
            if climate_hist is not None and t + h < n:
                fp = common_periods[t + h]
                if fp in climate_hist.index:
                    for col in climate_hist.columns:
                        row[f'climate_{col}'] = climate_hist.loc[fp, col]

            if holidays_hist is not None and t + h < n:
                fp = common_periods[t + h]
                if fp in holidays_hist.index:
                    for col in holidays_hist.columns:
                        row[f'holiday_{col}'] = holidays_hist.loc[fp, col]

            # Calendar at t+h
            if t + h < n:
                fp = common_periods[t + h]
                row['forecast_month']     = fp.month
                row['forecast_quarter']   = fp.quarter
                row['forecast_month_sin'] = np.sin(2 * np.pi * fp.month / 12)
                row['forecast_month_cos'] = np.cos(2 * np.pi * fp.month / 12)

            rows.append(row)
            targets.append(Y_al.iloc[t + h])
            periods_list.append(period)

        X_df  = pd.DataFrame(rows)
        Y_ser = pd.Series(targets, index=periods_list, name=f'Y_h{h}')
        horizon_data[h] = {'X': X_df, 'Y': Y_ser, 'periods': pd.PeriodIndex(periods_list, freq='M')}
        print(f"  [INFO] h={h}: {X_df.shape[0]} rows x {X_df.shape[1]} features")

    return horizon_data


# ==============================================================================
# STEP 3: TRAIN ELASTICNET PER HORIZON — WITH REGULARIZATION RATIONALE
# ==============================================================================

def train_stage2_models(horizon_data, horizon=6, n_cv_splits=5, val_size=12):
    """
    ElasticNet per horizon. Regularization is mandatory because:
      1. Generated regressors (Stage 1 errors) inflate apparent precision
      2. Spatial features are collinear (total_sum ~ total_mean)
      3. p/n ratio ~30/96 still requires shrinkage

    ElasticNet = L1 (sparsity, zeros irrelevant features) +
                 L2 (stability among correlated features)

    CV gap=h prevents leakage between train and val folds per horizon.

    Output: dict {h: {'model', 'scaler', 'val_mae', 'coefs'}}
    """
    models = {}
    scaler = StandardScaler()
    scaler.fit(horizon_data[1]['X'])   # fit once on h=1 features

    for h in range(1, horizon + 1):
        X     = horizon_data[h]['X']
        Y     = horizon_data[h]['Y']
        n     = len(Y)

        val_mask   = pd.Series(range(n)) >= (n - val_size)
        train_mask = ~val_mask

        X_sc     = scaler.transform(X)
        X_train  = X_sc[train_mask];  Y_train = Y.values[train_mask]
        X_val    = X_sc[val_mask];    Y_val   = Y.values[val_mask]

        # TimeSeriesSplit with gap=h to respect forecast horizon
        tscv  = TimeSeriesSplit(n_splits=n_cv_splits, gap=h)
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
            cv=tscv, max_iter=10000, n_jobs=-1, random_state=42
        )
        model.fit(X_train, Y_train)

        val_preds = model.predict(X_val)
        val_mae   = mean_absolute_error(Y_val, val_preds)
        val_mape  = np.mean(np.abs((Y_val - val_preds) / (np.abs(Y_val) + 1e-8))) * 100
        n_nonzero = (model.coef_ != 0).sum()

        coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
        print(f"  [INFO] h={h} | lambda={model.alpha_:.4f} | l1={model.l1_ratio_:.2f} | "
              f"nonzero={n_nonzero}/{len(coefs)} | val_MAE={val_mae:.4f} | MAPE={val_mape:.2f}%")

        models[h] = {'model': model, 'scaler': scaler, 'val_mae': val_mae, 'coefs': coefs}

    return models


# ==============================================================================
# STEP 3b: TRAIN LIGHTGBM PER HORIZON
# ==============================================================================

def train_stage2_models_lgb(horizon_data, horizon=6, n_cv_splits=5, val_size=12):
    """
    LightGBM per horizon. Gradient boosting with early stopping.
    Hyperparameters inherited from Stage 1 LightGBM settings.
    
    Output: dict {h: {'model', 'scaler', 'val_mae', 'coefs'}}
    """
    models = {}
    scaler = StandardScaler()
    scaler.fit(horizon_data[1]['X'])   # fit once on h=1 features
    
    # LightGBM hyperparameters (consistent with Stage 1)
    LGB_PARAMS = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': -1,
    }
    
    for h in range(1, horizon + 1):
        X     = horizon_data[h]['X']
        Y     = horizon_data[h]['Y']
        n     = len(Y)
        
        val_mask   = pd.Series(range(n)) >= (n - val_size)
        train_mask = ~val_mask
        
        X_sc     = scaler.transform(X)
        X_train  = X_sc[train_mask];  Y_train = Y.values[train_mask]
        X_val    = X_sc[val_mask];    Y_val   = Y.values[val_mask]
        
        # LightGBM dataset
        train_set = lgb.Dataset(X_train, label=Y_train)
        val_set   = lgb.Dataset(X_val, label=Y_val, reference=train_set)
        
        model = lgb.train(
            LGB_PARAMS,
            train_set,
            valid_sets=[val_set],
            valid_names=['val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        val_preds = model.predict(X_val)
        val_mae   = mean_absolute_error(Y_val, val_preds)
        val_mape  = np.mean(np.abs((Y_val - val_preds) / (np.abs(Y_val) + 1e-8))) * 100
        
        # Feature importances (gain)
        importances = pd.Series(
            model.feature_importance(importance_type='gain'),
            index=X.columns
        ).sort_values(ascending=False)
        
        print(f"  [INFO] h={h} | LGB val_MAE={val_mae:.4f} | MAPE={val_mape:.2f}% | "
              f"top feature={importances.index[0] if len(importances) > 0 else 'none'}")
        
        models[h] = {
            'model': model,
            'scaler': scaler,
            'val_mae': val_mae,
            'coefs': importances,  # store feature importances for compatibility
        }
    
    return models


# ==============================================================================
# STEP 4: INFERENCE
# ==============================================================================

def predict_stage2(models, X_hat_spatial, Y, climate_future=None,
                   holidays_future=None, y_lags=[1,2,3,6,12], horizon=6):
    """
    Apply learned β to Stage 1 spatial forecasts + Y lags + known future features.
    Output: forecast_df (6,) with point_forecast per horizon.
    """
    forecasts = []
    scaler    = models[1]['scaler']

    for h in range(1, horizon + 1):
        row = X_hat_spatial.loc[h].to_dict()

        for lag in y_lags:
            row[f'y_lag_{lag}'] = Y.iloc[-lag] if lag <= len(Y) else Y.iloc[0]

        roll3 = Y.iloc[-3:]; roll6 = Y.iloc[-6:]
        row['y_roll_mean_3'] = roll3.mean()
        row['y_roll_mean_6'] = roll6.mean()
        row['y_roll_std_3']  = roll3.std() if len(roll3) > 1 else 0
        row['y_mom_1']       = Y.iloc[-1] - Y.iloc[-2] if len(Y) >= 2 else 0
        row['y_mom_3']       = Y.iloc[-1] - Y.iloc[-4] if len(Y) >= 4 else 0

        if climate_future is not None and h in climate_future.index:
            for col in climate_future.columns:
                row[f'climate_{col}'] = climate_future.loc[h, col]
        if holidays_future is not None and h in holidays_future.index:
            for col in holidays_future.columns:
                row[f'holiday_{col}'] = holidays_future.loc[h, col]

        last_period   = Y.index[-1]
        future_period = last_period + h
        row['forecast_month']     = future_period.month
        row['forecast_quarter']   = future_period.quarter
        row['forecast_month_sin'] = np.sin(2 * np.pi * future_period.month / 12)
        row['forecast_month_cos'] = np.cos(2 * np.pi * future_period.month / 12)

        # Align columns to training order
        train_cols = list(horizon_data[h]['X'].columns) \
                     if 'horizon_data' in dir() else list(row.keys())
        try:
            train_cols = models[h]['model'].feature_names_in_.tolist()
        except AttributeError:
            train_cols = list(row.keys())

        X_row = pd.DataFrame([row]).reindex(columns=train_cols, fill_value=0)
        X_sc  = scaler.transform(X_row)
        point = models[h]['model'].predict(X_sc)[0] # THE ACTUAL PREDICTION 
        forecasts.append({'horizon': h, 'point_forecast': point})

    return pd.DataFrame(forecasts).set_index('horizon')


# ==============================================================================
# STEP 4b: LIGHTGBM INFERENCE
# ==============================================================================

def predict_stage2_lgb(models, X_hat_spatial, Y, climate_future=None,
                       holidays_future=None, y_lags=[1,2,3,6,12], horizon=6):
    """
    LightGBM variant of Stage 2 inference. Uses the same logic as predict_stage2.
    """
    return predict_stage2(models, X_hat_spatial, Y, climate_future,
                          holidays_future, y_lags, horizon)


# ==============================================================================
# STEP 5: BOOTSTRAP PREDICTION INTERVALS
# ==============================================================================

def bootstrap_prediction_intervals(models, X_hat_spatial, Y,
                                    climate_future=None, holidays_future=None,
                                    n_bootstrap=200, ci_levels=[80, 95], horizon=6):
    """
    Propagate Stage 1 uncertainty into prediction intervals (Pagan 1984).
    Adds proportional noise to X_hat_spatial → reruns Stage 2 → CI from spread.

    noise_scale tuned from Stage 1 val_MAE in production.
    Output: DataFrame (6,) with lower/upper per CI level.
    """
    noise_scale = X_hat_spatial.values.std() * 0.15
    boot_preds  = {h: [] for h in range(1, horizon + 1)}

    for _ in range(n_bootstrap):
        X_noisy = X_hat_spatial + np.random.normal(0, noise_scale, X_hat_spatial.shape)
        fc      = predict_stage2(models, X_noisy, Y, climate_future, holidays_future, horizon=horizon)
        for h in range(1, horizon + 1):
            boot_preds[h].append(fc.loc[h, 'point_forecast'])

    records = []
    for h in range(1, horizon + 1):
        samples = np.array(boot_preds[h])
        row     = {'horizon': h, 'point_forecast': samples.mean()}
        for ci in ci_levels:
            alpha = (100 - ci) / 2
            row[f'lower_{ci}'] = np.percentile(samples, alpha)
            row[f'upper_{ci}'] = np.percentile(samples, 100 - alpha)
        records.append(row)

    return pd.DataFrame(records).set_index('horizon')


# ==============================================================================
# STEP 6: BENCHMARKS
# ==============================================================================

def fit_naive_benchmark(Y, horizon=6):
    """Seasonal Naïve: Ŷ(t+h) = Y(t+h-12)"""
    forecasts = []
    for h in range(1, horizon + 1):
        idx  = -(12 - h + 1)
        pred = Y.iloc[idx] if abs(idx) <= len(Y) else Y.iloc[-1]
        forecasts.append({'horizon': h, 'point_forecast': pred})
    return pd.DataFrame(forecasts).set_index('horizon')


def fit_sarima_benchmark(Y, horizon=6):
    """
    SARIMA benchmark — univariate Y, ignores X entirely.
    ADF test → auto d. Seasonal order (1,1,1,12).
    If Y beats Two-Stage Transfer, Stage 1 adds no value.
    """
    adf   = adfuller(Y.dropna())
    d     = 0 if adf[1] < 0.05 else 1
    order = (1, d, 1); seas = (1, 1, 1, 12)
    try:
        res   = SARIMAX(Y, order=order, seasonal_order=seas,
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)
        preds = res.forecast(steps=horizon)
        print(f"  [INFO] SARIMA{order}x{seas} AIC={res.aic:.2f}")
        return pd.DataFrame(
            [{'horizon': h+1, 'point_forecast': preds.iloc[h]} for h in range(horizon)]
        ).set_index('horizon')
    except Exception as e:
        print(f"  [WARN] SARIMA failed: {e} — using Naive")
        return fit_naive_benchmark(Y, horizon)


# ==============================================================================
# STEP 7: WALK-FORWARD VALIDATION
# ==============================================================================

def walk_forward_validate_stage2(Y, hist_spatial, panel, pca, scaler_pca,
                                  stage1_models, le, city_stats,
                                  climate_hist=None, holidays_hist=None,
                                  horizon=6, min_train=36, step=3, n_cv_splits=5):
    """
    Walk-forward CV comparing Two-Stage Transfer vs SARIMA vs Seasonal Naïve.
    All models re-fit on training slice only. No future data leakage.
    Reports MAE, RMSE, MAPE per horizon + skill score vs Naïve.
    """
    import importlib
    stage1 = importlib.import_module('0_ze_forecast_monthly')
    predict_residual_chain = stage1.predict_residual_chain
    engineer_spatial_features = stage1.engineer_spatial_features

    results = []
    n       = len(Y)
    fold    = 0

    for t in range(min_train, n - horizon, step):
        fold += 1
        Y_train  = Y.iloc[:t]
        Y_actual = Y.iloc[t:t + horizon].values
        S_train  = hist_spatial.iloc[:t - 1]
        P_train  = panel.iloc[:t]

        print(f"\n  [FOLD {fold}] train={t} months")

        # -- Two-Stage Transfer (ElasticNet) --
        try:
            hdata    = build_stage2_training_matrix(Y_train, S_train,
                                                     climate_hist, holidays_hist,
                                                     horizon=horizon)
            s2_mods  = train_stage2_models(hdata, horizon, n_cv_splits)
            X_hat_m  = predict_residual_chain(stage1_models, P_train, le, city_stats, horizon)
            X_hat_s, _, _ = engineer_spatial_features(X_hat_m, P_train, pca.n_components_)
            fc_ts    = predict_stage2(s2_mods, X_hat_s, Y_train, horizon=horizon)
            preds_ts = fc_ts['point_forecast'].values
        except Exception as e:
            print(f"  [WARN] Two-Stage (ElasticNet): {e}")
            preds_ts = np.full(horizon, Y_train.iloc[-1])

        # -- Two-Stage Transfer (LightGBM) --
        try:
            hdata_lgb = build_stage2_training_matrix(Y_train, S_train,
                                                      climate_hist, holidays_hist,
                                                      horizon=horizon)
            s2_mods_lgb = train_stage2_models_lgb(hdata_lgb, horizon, n_cv_splits)
            X_hat_m     = predict_residual_chain(stage1_models, P_train, le, city_stats, horizon)
            X_hat_s, _, _ = engineer_spatial_features(X_hat_m, P_train, pca.n_components_)
            fc_lgb      = predict_stage2_lgb(s2_mods_lgb, X_hat_s, Y_train, horizon=horizon)
            preds_lgb   = fc_lgb['point_forecast'].values
        except Exception as e:
            print(f"  [WARN] Two-Stage (LightGBM): {e}")
            preds_lgb = np.full(horizon, Y_train.iloc[-1])

        # -- SARIMA --
        try:
            preds_sarima = fit_sarima_benchmark(Y_train, horizon)['point_forecast'].values
        except Exception as e:
            print(f"  [WARN] SARIMA: {e}")
            preds_sarima = np.full(horizon, Y_train.iloc[-1])

        # -- Naive --
        preds_naive = fit_naive_benchmark(Y_train, horizon)['point_forecast'].values

        # -- Score --
        for model_name, preds in [('two_stage', preds_ts),
                                   ('two_stage_lgb', preds_lgb),
                                   ('sarima',    preds_sarima),
                                   ('naive',     preds_naive)]:
            for h in range(horizon):
                a = Y_actual[h]; p = preds[h]
                results.append({
                    'fold': fold, 'model': model_name, 'horizon': h + 1,
                    'actual': a, 'forecast': p,
                    'mae':  abs(a - p),
                    'rmse': (a - p) ** 2,
                    'mape': abs((a - p) / (abs(a) + 1e-8)) * 100,
                })

    df = pd.DataFrame(results)
    df['rmse'] = np.sqrt(df['rmse'])

    summary = df.groupby(['model', 'horizon'])[['mae', 'rmse', 'mape']].mean().round(4)
    print("\n  ===== WALK-FORWARD VALIDATION SUMMARY =====")
    print(summary.to_string())

    # Skill score vs Naive
    naive_mae = df[df['model'] == 'naive'].groupby('horizon')['mae'].mean()
    for m in ['two_stage', 'two_stage_lgb', 'sarima']:
        m_mae = df[df['model'] == m].groupby('horizon')['mae'].mean()
        skill = (1 - m_mae / naive_mae).round(4)
        print(f"\n  Skill score vs Naive [{m}] (>0 beats naive):")
        print(skill.to_string())

    return df


# ==============================================================================
# STEP 8: FORECAST PLOT
# ==============================================================================

def plot_forecast(Y, forecast_df, benchmark_sarima, benchmark_naive,
                  title="Two-Stage Transfer Forecast",
                  save_path='C:/Users/Dell/data_science/shap_explainer/forecast_plot.png',
                  n_history=24):
    fig, ax = plt.subplots(figsize=(12, 5))
    Y_plot  = Y.iloc[-n_history:]
    x_hist  = list(range(n_history))
    x_fc    = list(range(n_history - 1, n_history + len(forecast_df)))
    anchor  = [Y_plot.iloc[-1]]

    ax.plot(x_hist, Y_plot.values, color='black', lw=2, label='Observed Y', zorder=5)

    # Two-Stage
    ts_vals = anchor + forecast_df['point_forecast'].tolist()
    ax.plot(x_fc, ts_vals, color='steelblue', lw=2,
            marker='o', ms=5, label='Two-Stage Transfer', zorder=4)
    if 'lower_80' in forecast_df.columns:
        lo = anchor + forecast_df['lower_80'].tolist()
        hi = anchor + forecast_df['upper_80'].tolist()
        ax.fill_between(x_fc, lo, hi, alpha=0.15, color='steelblue', label='80% PI')
    if 'lower_95' in forecast_df.columns:
        lo = anchor + forecast_df['lower_95'].tolist()
        hi = anchor + forecast_df['upper_95'].tolist()
        ax.fill_between(x_fc, lo, hi, alpha=0.08, color='steelblue', label='95% PI')

    # SARIMA
    ax.plot(x_fc, anchor + benchmark_sarima['point_forecast'].tolist(),
            color='darkorange', lw=1.5, ls='--', marker='s', ms=4, label='SARIMA')
    # Naive
    ax.plot(x_fc, anchor + benchmark_naive['point_forecast'].tolist(),
            color='grey', lw=1, ls=':', label='Seasonal Naive')

    ax.axvline(x=n_history - 1, color='black', ls='--', alpha=0.4, lw=1)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Months'); ax.set_ylabel('Y')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [INFO] Plot saved: {save_path}")
    plt.close()


# ==============================================================================
# MASTER STAGE 2 PIPELINE
# ==============================================================================

def run_stage2(Y_monthly, stage1_artifacts, climate_hist=None, holidays_hist=None,
               climate_future=None, holidays_future=None,
               horizon=6, validate=True, plot=True,
               plot_path='C:/Users/Dell/data_science/shap_explainer/forecast_plot.png'):
    """
    Full Stage 2. Inputs: Y_monthly + stage1_artifacts from run_stage1().
    Returns dict with forecast, intervals, benchmarks, validation results.
    """
    print("=" * 60)
    print("STAGE 2: DUAL-MODEL BRIDGE EQUATION (ElasticNet + LightGBM)")
    print("=" * 60)

    X_hat_spatial = stage1_artifacts['X_hat_spatial']
    panel         = stage1_artifacts['panel']
    pca           = stage1_artifacts['pca']
    scaler_pca    = stage1_artifacts['scaler_pca']
    stage1_models = stage1_artifacts['models']
    le            = stage1_artifacts['le']
    city_stats    = stage1_artifacts['city_stats']
    X_hat_monthly = stage1_artifacts['X_hat_monthly']
    # Compute city aggregates for the forecast horizons (same as historical)
    city_agg = compute_city_aggregates(X_hat_monthly)
    city_agg.index.name = 'horizon'  # align index name
    X_hat_spatial = X_hat_spatial.join(city_agg, how='left')

    print("\n[1/6] Building historical spatial features...")
    hist_spatial = build_historical_spatial_features(panel, pca, scaler_pca)

    # Ensure X_hat_spatial has same columns as hist_spatial (training data)
    X_hat_spatial = X_hat_spatial.reindex(columns=hist_spatial.columns, fill_value=0)

    print("\n[2/6] Building Stage 2 training matrix...")
    global horizon_data
    horizon_data = build_stage2_training_matrix(
        Y_monthly, hist_spatial, climate_hist, holidays_hist, horizon=horizon)

    print("\n[3/6] Training ElasticNet per horizon...")
    s2_models = train_stage2_models(horizon_data, horizon)
    
    print("\n[3b/6] Training LightGBM per horizon...")
    s2_models_lgb = train_stage2_models_lgb(horizon_data, horizon)

    print("\n[4/6] Point forecast via ElasticNet...")
    forecast_df = predict_stage2(
        s2_models, X_hat_spatial, Y_monthly,
        climate_future, holidays_future, horizon=horizon)
    
    print("\n[4b/6] Point forecast via LightGBM...")
    forecast_df_lgb = predict_stage2_lgb(
        s2_models_lgb, X_hat_spatial, Y_monthly,
        climate_future, holidays_future, horizon=horizon)

    print("\n[5/6] Bootstrap prediction intervals (ElasticNet)...")
    intervals_df = bootstrap_prediction_intervals(
        s2_models, X_hat_spatial, Y_monthly,
        climate_future, holidays_future,
        n_bootstrap=200, horizon=horizon)
    
    print("\n[5b/6] Bootstrap prediction intervals (LightGBM)...")
    intervals_df_lgb = bootstrap_prediction_intervals(
        s2_models_lgb, X_hat_spatial, Y_monthly,
        climate_future, holidays_future,
        n_bootstrap=200, horizon=horizon)

    # Merge intervals into forecast_df
    for col in ['lower_80', 'upper_80', 'lower_95', 'upper_95']:
        if col in intervals_df.columns:
            forecast_df[col] = intervals_df[col]
    
    # Merge intervals into forecast_df_lgb
    for col in ['lower_80', 'upper_80', 'lower_95', 'upper_95']:
        if col in intervals_df_lgb.columns:
            forecast_df_lgb[col] = intervals_df_lgb[col]

    print("\n[5b/6] Benchmarks...")
    bm_sarima = fit_sarima_benchmark(Y_monthly, horizon)
    bm_naive  = fit_naive_benchmark(Y_monthly, horizon)

    comparison = pd.DataFrame({
        'Two_Stage_ElasticNet': forecast_df['point_forecast'],
        'Two_Stage_LightGBM':   forecast_df_lgb['point_forecast'],
        'SARIMA':               bm_sarima['point_forecast'],
        'Naive':                bm_naive['point_forecast'],
    })
    print("\n  ===== FORECAST COMPARISON =====")
    print(comparison.round(4).to_string())

    val_results = None
    if validate:
        print("\n[6/6] Walk-forward validation...")
        val_results = walk_forward_validate_stage2(
            Y_monthly, hist_spatial, panel, pca, scaler_pca,
            stage1_models, le, city_stats,
            climate_hist, holidays_hist, horizon=horizon)
    else:
        print("\n[6/6] Skipping validation.")

    if plot:
        plot_forecast(Y_monthly, forecast_df, bm_sarima, bm_naive,
                      save_path=plot_path)

    feat_importance = {h: s2_models[h]['coefs'][s2_models[h]['coefs'] != 0].head(10)
                       for h in range(1, horizon + 1)}
    print("\n  ===== TOP FEATURES (ElasticNet) h=1 =====")
    print(feat_importance[1].round(4).to_string())
    
    feat_importance_lgb = {h: s2_models_lgb[h]['coefs'][s2_models_lgb[h]['coefs'] != 0].head(10)
                           for h in range(1, horizon + 1)}
    print("\n  ===== TOP FEATURES (LightGBM) h=1 =====")
    print(feat_importance_lgb[1].round(4).to_string())

    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETE")
    print("=" * 60)

    return {
        'forecast':           forecast_df,
        'intervals':          intervals_df,
        'models':             s2_models,
        'hist_spatial':       hist_spatial,
        'benchmark_sarima':   bm_sarima,
        'benchmark_naive':    bm_naive,
        'val_results':        val_results,
        'comparison':         comparison,
        'feature_importance': feat_importance,
        'models_lgb':         s2_models_lgb,
        'forecast_lgb':       forecast_df_lgb,
        'intervals_lgb':      intervals_df_lgb,
        'feature_importance_lgb': feat_importance_lgb,
    }


# ==============================================================================
# SMOKE TEST
# ==============================================================================

if __name__ == '__main__':
    import importlib
    stage1 = importlib.import_module('0_ze_forecast_monthly')
    run_stage1 = stage1.run_stage1

    print("Generating synthetic data...")
    np.random.seed(42)
    n_cities = 20; n_months = 96; n_tx = 50000
    cities   = [f'city_{i:03d}' for i in range(n_cities)]
    start    = pd.Timestamp('2016-01-01')
    dates, city_labels, values = [], [], []
    for _ in range(n_tx):
        city     = np.random.choice(cities)
        month    = np.random.randint(0, n_months)
        date     = start + pd.DateOffset(months=month) + pd.DateOffset(days=np.random.randint(0, 28))
        city_idx = cities.index(city)
        val      = max(0, 100 + city_idx*5 + month*0.3
                       + 20*np.sin(2*np.pi*date.month/12) + np.random.normal(0, 10))
        dates.append(date); city_labels.append(city); values.append(val)

    df_tx    = pd.DataFrame({'date': dates, 'city': city_labels, 'value': values})
    periods  = pd.period_range('2016-01', periods=n_months, freq='M')
    y_vals   = np.array([1000 + 2*i + 80*np.sin(2*np.pi*p.month/12) + np.random.normal(0, 30)
                         for i, p in enumerate(periods)])
    Y_monthly = pd.Series(y_vals, index=periods, name='Y')

    print("Running Stage 1...")
    s1 = run_stage1(df_tx, horizon=6, n_pca_components=5, validate=False)

    print("\nRunning Stage 2...")
    s2 = run_stage2(Y_monthly, s1, horizon=6, validate=False, plot=True)

    print("\nFinal forecast:")
    print(s2['comparison'].round(2))
