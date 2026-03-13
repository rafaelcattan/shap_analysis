"""
TWO-STAGE TRANSFER FORECASTING  -  STAGE 1
==========================================
Purpose : Forecast n city-level X series from transaction data
          forward h=1->6 months using Residual Chaining GlobalLightGBM
Output  : X_hat_monthly    (6, n_cities)     -  level forecasts per city per horizon
          X_hat_spatial    (6, n_features)   -  spatial engineered features for Stage 2
          model_artifacts  dict              -  models + scalers for reproducibility

Literature:
  - Lag features:        Box e Jenkins (1970) ARIMA framework
  - Calendar features:   Cleveland et al. (1990) STL decomposition
  - Rolling statistics:  Hyndman and Athanasopoulos (2021) Forecasting: PandP
  - Fourier features:    Harvey and Shephard (1993) structural time series
  - Global models:       Montero-Manso and Hyndman (2021) "Principles and Algorithms
                         for Forecasting Groups of Time Series"
  - Residual chaining:   Chevillon (2007) direct multi-step estimation
  - Spatial features:    Stock and Watson (2002) diffusion indexes (FAVAR precursor)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

warnings.filterwarnings('ignore')


# ==============================================================================
# STEP 1: AGGREGATE TRANSACTIONS -> MONTHLY CITY PANEL
# ==============================================================================

def aggregate_transactions_to_monthly(
    df: pd.DataFrame,
    date_col:   str = 'date',
    city_col:   str = 'city',
    value_col:  str = 'value',
    agg_func:   str = 'sum'          # 'sum' for volumes, 'mean' for rates/prices
) -> pd.DataFrame:
    """
    Convert raw transaction-level data to monthly city panel.

    Input:
        df: raw transactions
            columns: [date, city, value, ...]
            shape:   (n_transactions, n_cols)

    Output:
        monthly panel in WIDE format
            index:   period (monthly)
            columns: city_1, city_2, ..., city_n
            shape:   (n_months, n_cities)

    Notes:
        - Missing city-months are forward-filled then backward-filled
          (city existed but had no transactions -> assume continuity)
        - Cities with >50% missing months are dropped
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['period']  = df[date_col].dt.to_period('M')

    # Aggregate transactions to monthly per city
    monthly = (
        df.groupby(['period', city_col])[value_col]
        .agg(agg_func)
        .reset_index()
    )

    # Pivot to wide format: rows=months, cols=cities
    panel = monthly.pivot(index='period', columns=city_col, values=value_col)
    panel = panel.sort_index()

    # Drop cities with excessive missing data (>50% months missing)
    missing_pct = panel.isna().mean()
    panel = panel.loc[:, missing_pct <= 0.5]
    dropped = (missing_pct > 0.5).sum()
    if dropped > 0:
        print(f"  [INFO] Dropped {dropped} cities with >50% missing months")

    # Fill remaining gaps: forward fill then backward fill
    panel = panel.ffill().bfill()

    print(f"  [INFO] Monthly panel: {panel.shape[0]} months x {panel.shape[1]} cities")
    return panel


# ==============================================================================
# STEP 2: FEATURE ENGINEERING  -  MONTHLY CITY PANEL
# ==============================================================================

def build_monthly_features(
    panel:        pd.DataFrame,   # (n_months, n_cities) wide format
    forecast_horizon: int = 6     # needed to avoid leakage in rolling features
) -> pd.DataFrame:
    """
    Convert wide monthly panel -> long stacked format with rich features.

    Feature groups (literature-backed):

    A. LAG FEATURES            Box & Jenkins (1970)
       lag_1, lag_2, lag_3     recent momentum
       lag_6                   semi-annual pattern
       lag_12                  annual seasonality
       lag_24                  2-year cycle

    B. ROLLING STATISTICS      Hyndman & Athanasopoulos (2021)
       rolling_mean_3/6/12     trend smoothing at multiple scales
       rolling_std_3/6/12      local volatility
       rolling_min/max_6       recent range
       rolling_skew_6          distributional asymmetry

    C. MOMENTUM / RATE OF CHANGE  Fama & French (1988) momentum literature
       mom_1                   1-month momentum (value - lag_1)
       mom_3                   3-month momentum
       mom_6                   6-month momentum
       pct_change_1/3/6        relative changes

    D. CALENDAR FEATURES       Cleveland et al. (1990) STL
       month                   month of year (1-12)
       quarter                 quarter (1-4)
       month_sin/cos           cyclical encoding of month
       quarter_sin/cos         cyclical encoding of quarter

    E. FOURIER FEATURES        Harvey & Shephard (1993)
       fourier_sin/cos k=1,2   harmonic seasonal decomposition
       captures smooth seasonality better than dummy variables

    F. TREND FEATURES          Henderson (1916) moving average filters
       trend_slope_6           linear trend slope over last 6 months
       trend_slope_12          linear trend slope over last 12 months
       deviation_from_trend    current value vs trend

    G. CITY IDENTITY           Montero-Manso & Hyndman (2021)
       city_id                 label-encoded city identifier
       city_mean               historical mean of city (static)
       city_std                historical std of city (static)
       city_cv                 coefficient of variation (std/mean)

    Output:
        long format DataFrame
        columns: [city_id, period, all_features..., target]
        shape:   (n_months * n_cities, n_features + 2)
    """

    cities  = panel.columns.tolist()
    n_cities = len(cities)

    # Label encode city identifiers
    le = LabelEncoder()
    le.fit(cities)

    # City-level static statistics (fit on full history  -  no leakage for static stats)
    city_stats = {
        city: {
            'city_mean': panel[city].mean(),
            'city_std':  panel[city].std(),
            'city_cv':   panel[city].std() / (panel[city].mean() + 1e-8),
        }
        for city in cities
    }

    records = []

    for city in cities:
        series   = panel[city].copy()
        n_months = len(series)
        periods  = series.index

        city_id_enc = le.transform([city])[0]

        for t in range(24, n_months):   # start at 24 to allow lag_24 without NaN

            period = periods[t]
            val    = series.iloc[t]

            # ── A. LAG FEATURES ──────────────────────────────────────────────
            lag_1  = series.iloc[t - 1]
            lag_2  = series.iloc[t - 2]
            lag_3  = series.iloc[t - 3]
            lag_6  = series.iloc[t - 6]
            lag_12 = series.iloc[t - 12]
            lag_24 = series.iloc[t - 24]

            # ── B. ROLLING STATISTICS ─────────────────────────────────────────
            roll3  = series.iloc[t-3:t]
            roll6  = series.iloc[t-6:t]
            roll12 = series.iloc[t-12:t]

            roll_mean_3   = roll3.mean()
            roll_mean_6   = roll6.mean()
            roll_mean_12  = roll12.mean()
            roll_std_3    = roll3.std()
            roll_std_6    = roll6.std()
            roll_std_12   = roll12.std()
            roll_min_6    = roll6.min()
            roll_max_6    = roll6.max()
            roll_range_6  = roll_max_6 - roll_min_6
            roll_skew_6   = float(stats.skew(roll6.values))

            # ── C. MOMENTUM / RATE OF CHANGE ─────────────────────────────────
            mom_1         = val - lag_1
            mom_3         = val - lag_3
            mom_6         = val - lag_6
            pct_change_1  = (val - lag_1)  / (abs(lag_1)  + 1e-8)
            pct_change_3  = (val - lag_3)  / (abs(lag_3)  + 1e-8)
            pct_change_6  = (val - lag_6)  / (abs(lag_6)  + 1e-8)
            accel         = mom_1 - (series.iloc[t-1] - series.iloc[t-2])  # Δ of Δ

            # ── D. CALENDAR FEATURES ─────────────────────────────────────────
            month_num     = period.month
            quarter_num   = period.quarter

            # Cyclical encoding  -  avoids discontinuity between Dec->Jan
            month_sin     = np.sin(2 * np.pi * month_num   / 12)
            month_cos     = np.cos(2 * np.pi * month_num   / 12)
            quarter_sin   = np.sin(2 * np.pi * quarter_num / 4)
            quarter_cos   = np.cos(2 * np.pi * quarter_num / 4)

            # ── E. FOURIER FEATURES ───────────────────────────────────────────
            # k=1,2 harmonics capture smooth annual seasonality
            # Better than month dummies: fewer parameters, smooth interpolation
            fourier_sin_1 = np.sin(2 * np.pi * 1 * month_num / 12)
            fourier_cos_1 = np.cos(2 * np.pi * 1 * month_num / 12)
            fourier_sin_2 = np.sin(2 * np.pi * 2 * month_num / 12)
            fourier_cos_2 = np.cos(2 * np.pi * 2 * month_num / 12)

            # ── F. TREND FEATURES ─────────────────────────────────────────────
            # OLS slope over rolling window = local trend strength and direction
            x6  = np.arange(6)
            x12 = np.arange(12)
            trend_slope_6  = float(np.polyfit(x6,  roll6.values,  1)[0])
            trend_slope_12 = float(np.polyfit(x12, roll12.values, 1)[0])
            deviation_from_trend = val - roll_mean_12   # above/below long-run mean

            # ── G. CITY IDENTITY ──────────────────────────────────────────────
            city_mean = city_stats[city]['city_mean']
            city_std  = city_stats[city]['city_std']
            city_cv   = city_stats[city]['city_cv']

            # Normalized value: where is this city relative to its own history?
            value_normalized = (val - city_mean) / (city_std + 1e-8)

            records.append({
                # identifiers
                'period':               period,
                'city':                 city,
                'city_id':              city_id_enc,
                # A. lags
                'lag_1':                lag_1,
                'lag_2':                lag_2,
                'lag_3':                lag_3,
                'lag_6':                lag_6,
                'lag_12':               lag_12,
                'lag_24':               lag_24,
                # B. rolling stats
                'roll_mean_3':          roll_mean_3,
                'roll_mean_6':          roll_mean_6,
                'roll_mean_12':         roll_mean_12,
                'roll_std_3':           roll_std_3,
                'roll_std_6':           roll_std_6,
                'roll_std_12':          roll_std_12,
                'roll_min_6':           roll_min_6,
                'roll_max_6':           roll_max_6,
                'roll_range_6':         roll_range_6,
                'roll_skew_6':          roll_skew_6,
                # C. momentum
                'mom_1':                mom_1,
                'mom_3':                mom_3,
                'mom_6':                mom_6,
                'pct_change_1':         pct_change_1,
                'pct_change_3':         pct_change_3,
                'pct_change_6':         pct_change_6,
                'acceleration':         accel,
                # D. calendar
                'month':                month_num,
                'quarter':              quarter_num,
                'month_sin':            month_sin,
                'month_cos':            month_cos,
                'quarter_sin':          quarter_sin,
                'quarter_cos':          quarter_cos,
                # E. fourier
                'fourier_sin_1':        fourier_sin_1,
                'fourier_cos_1':        fourier_cos_1,
                'fourier_sin_2':        fourier_sin_2,
                'fourier_cos_2':        fourier_cos_2,
                # F. trend
                'trend_slope_6':        trend_slope_6,
                'trend_slope_12':       trend_slope_12,
                'deviation_from_trend': deviation_from_trend,
                # G. city identity
                'city_mean':            city_mean,
                'city_std':             city_std,
                'city_cv':              city_cv,
                'value_normalized':     value_normalized,
                # target
                'value':                val,
            })

    features_df = pd.DataFrame(records)
    print(f"  [INFO] Feature matrix: {features_df.shape[0]} rows x {features_df.shape[1]} cols")
    print(f"  [INFO] Features per city: {len(records) // n_cities} time steps")
    return features_df, le, city_stats


# ==============================================================================
# STEP 3: RESIDUAL CHAINING  -  TRAIN 6 GLOBAL LIGHTGBM MODELS
# ==============================================================================

# Feature columns used as model input (excludes identifiers and target)
FEATURE_COLS = [
    'city_id',
    # lags
    'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'lag_24',
    # rolling
    'roll_mean_3', 'roll_mean_6', 'roll_mean_12',
    'roll_std_3', 'roll_std_6', 'roll_std_12',
    'roll_min_6', 'roll_max_6', 'roll_range_6', 'roll_skew_6',
    # momentum
    'mom_1', 'mom_3', 'mom_6',
    'pct_change_1', 'pct_change_3', 'pct_change_6', 'acceleration',
    # calendar
    'month', 'quarter',
    'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
    # fourier
    'fourier_sin_1', 'fourier_cos_1', 'fourier_sin_2', 'fourier_cos_2',
    # trend
    'trend_slope_6', 'trend_slope_12', 'deviation_from_trend',
    # city identity
    'city_mean', 'city_std', 'city_cv', 'value_normalized',
]

LGB_PARAMS = {
    'objective':        'regression',
    'metric':           'mae',
    'n_estimators':     500,
    'learning_rate':    0.05,
    'max_depth':        5,
    'num_leaves':       31,
    'min_child_samples':20,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'n_jobs':           -1,
    'verbose':          -1,
    'random_state':     42,
}


def build_residual_chain_targets(
    features_df: pd.DataFrame,
    panel:       pd.DataFrame,
    horizon:     int = 6
) -> dict:
    """
    Build training targets for each horizon h in 1->6 using residual chaining.

    Residual chaining targets:
        h=1: absolute level    X(t+1)
        h=2: increment         X(t+2) - X(t+1)
        h=3: increment         X(t+3) - X(t+2)
        ...
        h=6: increment         X(t+6) - X(t+5)

    For each horizon, merge the appropriate target into features_df.

    Output:
        dict {h: DataFrame with features + target_h column}
        target_h = absolute level for h=1, increment for h>1
    """
    horizon_data = {}

    for h in range(1, horizon + 1):
        df_h = features_df.copy()

        # Build target series per city
        target_rows = []
        for city in panel.columns:
            city_df = df_h[df_h['city'] == city].copy()
            city_series = panel[city]
            periods = city_series.index

            for idx, row in city_df.iterrows():
                period = row['period']
                t_loc  = list(periods).index(period)

                # Ensure we have t+h in the panel
                if t_loc + h >= len(periods):
                    continue

                val_t_plus_h = city_series.iloc[t_loc + h]

                if h == 1:
                    # h=1: predict absolute level
                    target = val_t_plus_h
                else:
                    # h>1: predict INCREMENT over previous horizon
                    val_t_plus_h_minus_1 = city_series.iloc[t_loc + h - 1]
                    target = val_t_plus_h - val_t_plus_h_minus_1

                target_rows.append({'period': period, 'city': city, f'target_h{h}': target})

        target_df = pd.DataFrame(target_rows)
        df_h = df_h.merge(target_df, on=['period', 'city'], how='inner')
        df_h = df_h.dropna(subset=FEATURE_COLS + [f'target_h{h}'])

        horizon_data[h] = df_h
        print(f"  [INFO] Horizon h={h}: {len(df_h)} training rows, "
              f"target={'level' if h==1 else 'increment'}")

    return horizon_data


def train_residual_chain_models(
    horizon_data: dict,
    horizon:      int = 6,
    val_months:   int = 12    # last N months used for early stopping validation
) -> dict:
    """
    Train one GlobalLightGBM per horizon using residual chaining targets.

    Each model is global across all cities  -  city_id is a feature.
    Training uses time-based split for early stopping:
        train: all months except last val_months
        val:   last val_months (early stopping only, not CV)

    Output:
        models dict {h: trained LGBModel}
    """
    models = {}

    for h in range(1, horizon + 1):
        df_h       = horizon_data[h]
        target_col = f'target_h{h}'

        # Time-based train/val split for early stopping
        all_periods  = sorted(df_h['period'].unique())
        val_periods  = all_periods[-val_months:]
        train_mask   = ~df_h['period'].isin(val_periods)
        val_mask     = df_h['period'].isin(val_periods)

        X_train = df_h.loc[train_mask, FEATURE_COLS]
        y_train = df_h.loc[train_mask, target_col]
        X_val   = df_h.loc[val_mask,   FEATURE_COLS]
        y_val   = df_h.loc[val_mask,   target_col]

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1)
            ]
        )

        val_preds = model.predict(X_val)
        val_mae   = mean_absolute_error(y_val, val_preds)
        print(f"  [INFO] h={h} | best_iteration={model.best_iteration_} "
              f"| val_MAE={val_mae:.4f} ({'level' if h==1 else 'increment'})")

        models[h] = model

    return models


# ==============================================================================
# STEP 4: INFERENCE  -  PREDICT FORWARD 6 MONTHS PER CITY
# ==============================================================================

def predict_residual_chain(
    models:      dict,           # {h: LGBModel}
    panel:       pd.DataFrame,   # (n_months, n_cities) full observed history
    le:          LabelEncoder,   # fitted city label encoder
    city_stats:  dict,           # city-level static statistics
    horizon:     int = 6
) -> pd.DataFrame:
    """
    Predict X_hat_monthly (6, n_cities) using residual chaining.

    For each horizon h:
        h=1: predict absolute level       X_hat[1]
        h=2: predict increment            Δ_hat[2] -> X_hat[2] = X_hat[1] + Δ_hat[2]
        h=3: predict increment            Δ_hat[3] -> X_hat[3] = X_hat[2] + Δ_hat[3]
        ...

    The feature vector for each prediction uses:
        - lag features from observed history + previously predicted values
        - calendar features for the forecast period
        - city identity features

    Output:
        X_hat_monthly: DataFrame (horizon, n_cities)
                       index: forecast horizon (1->6)
                       columns: city names
    """
    cities    = panel.columns.tolist()
    n_months  = len(panel)
    periods   = panel.index

    # Determine forecast periods (next 6 months after panel end)
    last_period      = periods[-1]
    forecast_periods = pd.period_range(
        start=last_period + 1, periods=horizon, freq='M'
    )

    # Initialize predictions store: observed + forecasts in buffer
    # buffer[city] = list of (period, value)  -  grows as we predict
    buffer = {city: list(panel[city].items()) for city in cities}

    X_hat = {}   # {h: {city: predicted_value}}

    for h in range(1, horizon + 1):
        X_hat[h] = {}
        forecast_period = forecast_periods[h - 1]

        # Build feature rows for all cities at this horizon
        rows = []
        for city in cities:
            city_buffer = pd.Series(
                [v for _, v in buffer[city]],
                index=pd.PeriodIndex([p for p, _ in buffer[city]], freq='M')
            )
            t = len(city_buffer)   # current position in buffer

            # Lag features  -  pull from buffer (observed + previously predicted)
            def get_lag(k):
                idx = t - k
                if idx < 0:
                    return city_buffer.iloc[0]   # boundary: repeat first value
                return city_buffer.iloc[idx]

            lag_1  = get_lag(1)
            lag_2  = get_lag(2)
            lag_3  = get_lag(3)
            lag_6  = get_lag(6)
            lag_12 = get_lag(12)
            lag_24 = get_lag(24)

            # Rolling statistics from buffer
            roll6  = city_buffer.iloc[max(0, t-6):t]
            roll12 = city_buffer.iloc[max(0, t-12):t]
            roll3  = city_buffer.iloc[max(0, t-3):t]

            roll_mean_3   = roll3.mean()
            roll_mean_6   = roll6.mean()
            roll_mean_12  = roll12.mean()
            roll_std_3    = roll3.std() if len(roll3) > 1 else 0
            roll_std_6    = roll6.std() if len(roll6) > 1 else 0
            roll_std_12   = roll12.std() if len(roll12) > 1 else 0
            roll_min_6    = roll6.min()
            roll_max_6    = roll6.max()
            roll_range_6  = roll_max_6 - roll_min_6
            roll_skew_6   = float(stats.skew(roll6.values)) if len(roll6) > 2 else 0

            # Momentum (using buffer values)
            current_val   = city_buffer.iloc[-1]
            mom_1         = current_val - lag_1
            mom_3         = current_val - lag_3
            mom_6         = current_val - lag_6
            pct_change_1  = (current_val - lag_1) / (abs(lag_1)  + 1e-8)
            pct_change_3  = (current_val - lag_3) / (abs(lag_3)  + 1e-8)
            pct_change_6  = (current_val - lag_6) / (abs(lag_6)  + 1e-8)
            accel         = mom_1 - (lag_1 - lag_2)

            # Calendar features for forecast period
            month_num     = forecast_period.month
            quarter_num   = forecast_period.quarter
            month_sin     = np.sin(2 * np.pi * month_num   / 12)
            month_cos     = np.cos(2 * np.pi * month_num   / 12)
            quarter_sin   = np.sin(2 * np.pi * quarter_num / 4)
            quarter_cos   = np.cos(2 * np.pi * quarter_num / 4)

            # Fourier
            fourier_sin_1 = np.sin(2 * np.pi * 1 * month_num / 12)
            fourier_cos_1 = np.cos(2 * np.pi * 1 * month_num / 12)
            fourier_sin_2 = np.sin(2 * np.pi * 2 * month_num / 12)
            fourier_cos_2 = np.cos(2 * np.pi * 2 * month_num / 12)

            # Trend
            x6  = np.arange(len(roll6))
            x12 = np.arange(len(roll12))
            trend_slope_6  = float(np.polyfit(x6,  roll6.values,  1)[0]) if len(roll6) >= 2 else 0
            trend_slope_12 = float(np.polyfit(x12, roll12.values, 1)[0]) if len(roll12) >= 2 else 0
            deviation_from_trend = current_val - roll_mean_12

            # City identity
            city_mean = city_stats[city]['city_mean']
            city_std  = city_stats[city]['city_std']
            city_cv   = city_stats[city]['city_cv']
            value_normalized = (current_val - city_mean) / (city_std + 1e-8)

            rows.append({
                'city':                 city,
                'city_id':              le.transform([city])[0],
                'lag_1':                lag_1,
                'lag_2':                lag_2,
                'lag_3':                lag_3,
                'lag_6':                lag_6,
                'lag_12':               lag_12,
                'lag_24':               lag_24,
                'roll_mean_3':          roll_mean_3,
                'roll_mean_6':          roll_mean_6,
                'roll_mean_12':         roll_mean_12,
                'roll_std_3':           roll_std_3,
                'roll_std_6':           roll_std_6,
                'roll_std_12':          roll_std_12,
                'roll_min_6':           roll_min_6,
                'roll_max_6':           roll_max_6,
                'roll_range_6':         roll_range_6,
                'roll_skew_6':          roll_skew_6,
                'mom_1':                mom_1,
                'mom_3':                mom_3,
                'mom_6':                mom_6,
                'pct_change_1':         pct_change_1,
                'pct_change_3':         pct_change_3,
                'pct_change_6':         pct_change_6,
                'acceleration':         accel,
                'month':                month_num,
                'quarter':              quarter_num,
                'month_sin':            month_sin,
                'month_cos':            month_cos,
                'quarter_sin':          quarter_sin,
                'quarter_cos':          quarter_cos,
                'fourier_sin_1':        fourier_sin_1,
                'fourier_cos_1':        fourier_cos_1,
                'fourier_sin_2':        fourier_sin_2,
                'fourier_cos_2':        fourier_cos_2,
                'trend_slope_6':        trend_slope_6,
                'trend_slope_12':       trend_slope_12,
                'deviation_from_trend': deviation_from_trend,
                'city_mean':            city_mean,
                'city_std':             city_std,
                'city_cv':              city_cv,
                'value_normalized':     value_normalized,
            })

        inference_df = pd.DataFrame(rows)
        X_inf        = inference_df[FEATURE_COLS]
        delta_preds  = models[h].predict(X_inf)   # increment or level

        # Reconstruct level from residual chain
        for i, city in enumerate(cities):
            delta = delta_preds[i]
            if h == 1:
                # h=1: model predicts absolute level directly
                level = delta
            else:
                # h>1: model predicts increment -> add to previous level
                level = X_hat[h - 1][city] + delta

            X_hat[h][city] = level

            # Append prediction to city buffer for next horizon's lag features
            buffer[city].append((forecast_period, level))

    # Assemble into DataFrame
    X_hat_monthly = pd.DataFrame(X_hat).T   # (horizon, n_cities)
    X_hat_monthly.index = range(1, horizon + 1)
    X_hat_monthly.index.name = 'horizon'

    return X_hat_monthly


# ==============================================================================
# STEP 5: ENGINEER SPATIAL FEATURES FROM X_HAT -> STAGE 2 INPUT
# ==============================================================================

def engineer_spatial_features(
    X_hat_monthly:  pd.DataFrame,    # (6, n_cities)  -  forecast level per city
    panel:          pd.DataFrame,    # (n_months, n_cities)  -  historical panel
    n_pca_components: int = 15
) -> pd.DataFrame:
    """
    Compress (6, n_cities) forecast matrix into (6, n_spatial_features)
    for use as Stage 2 input.

    Spatial features per forecast horizon h:

    LEVEL FEATURES        Stock & Watson (2002)
        total_sum         sum across all cities
        total_mean        mean across all cities
        total_median      median (robust to outliers)

    BREADTH FEATURES      McClellan (1969) market breadth literature
        breadth_above_hist_mean   % cities above their own historical mean
        breadth_positive_mom      % cities with positive month-on-month change
        breadth_above_trend       % cities above their 6-month trend

    DISPERSION FEATURES   Loungani et al. (2001) cross-sectional dispersion
        spatial_std       std across cities (how concentrated the activity is)
        spatial_cv        coefficient of variation
        spatial_skew      distributional asymmetry across cities
        spatial_iqr       interquartile range (robust dispersion)

    CONCENTRATION         Herfindahl-Hirschman Index concept
        top10_share       share of top 10 cities in total
        top10_sum         absolute sum of top 10 cities
        gini_coeff        Gini coefficient across city forecasts

    PCA FACTORS           Bernanke et al. (2005) FAVAR
        pca_factor_1..k   spatial configuration compressed to k factors

    Output:
        spatial_df: (6, n_spatial_features)
        index: horizon (1->6)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cities    = panel.columns.tolist()
    n_cities  = len(cities)

    # Historical city means for breadth features
    hist_means  = panel.mean()
    hist_6m_avg = panel.iloc[-6:].mean()

    # Fit PCA on historical panel for spatial factor extraction
    # panel.values: (n_months, n_cities) -> PCA across city dimension
    # Each row = one month's snapshot across all cities
    scaler_pca = StandardScaler()
    panel_scaled = scaler_pca.fit_transform(panel.values)  # (n_months, n_cities)
    n_components = min(n_pca_components, n_cities - 1, panel_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components)
    pca.fit(panel_scaled)

    def gini(arr):
        arr = np.sort(np.abs(arr))
        n   = len(arr)
        idx = np.arange(1, n + 1)
        return (2 * np.sum(idx * arr) / (n * np.sum(arr) + 1e-8)) - (n + 1) / n

    spatial_records = []

    for h in X_hat_monthly.index:
        city_vals = X_hat_monthly.loc[h]   # Series (n_cities,)
        vals      = city_vals.values
        n         = len(vals)

        # Previous horizon values for momentum (h=1 uses last observed month)
        if h == 1:
            prev_vals = panel.iloc[-1].values
        else:
            prev_vals = X_hat_monthly.loc[h - 1].values

        # ── LEVEL ────────────────────────────────────────────────────────────
        total_sum    = vals.sum()
        total_mean   = vals.mean()
        total_median = np.median(vals)

        # ── BREADTH ──────────────────────────────────────────────────────────
        breadth_above_hist  = (vals > hist_means.values).mean()
        breadth_pos_mom     = (vals > prev_vals).mean()
        breadth_above_trend = (vals > hist_6m_avg.values).mean()

        # ── DISPERSION ───────────────────────────────────────────────────────
        spatial_std  = vals.std()
        spatial_cv   = spatial_std / (abs(total_mean) + 1e-8)
        spatial_skew = float(stats.skew(vals))
        spatial_iqr  = float(np.percentile(vals, 75) - np.percentile(vals, 25))

        # ── CONCENTRATION ────────────────────────────────────────────────────
        sorted_desc  = np.sort(vals)[::-1]
        top10_idx    = min(10, n)
        top10_sum    = sorted_desc[:top10_idx].sum()
        top10_share  = top10_sum / (total_sum + 1e-8)
        gini_coeff   = gini(vals)

        # ── PCA SPATIAL FACTORS ───────────────────────────────────────────────
        # scaler_pca fit on (n_months, n_cities) -> transform expects (1, n_cities)
        vals_norm   = scaler_pca.transform(vals.reshape(1, -1))        # (1, n_cities)
        pca_factors = pca.transform(vals_norm)[0]                       # (n_components,)

        row = {
            'horizon':              h,
            # level
            'total_sum':            total_sum,
            'total_mean':           total_mean,
            'total_median':         total_median,
            # breadth
            'breadth_above_hist':   breadth_above_hist,
            'breadth_pos_mom':      breadth_pos_mom,
            'breadth_above_trend':  breadth_above_trend,
            # dispersion
            'spatial_std':          spatial_std,
            'spatial_cv':           spatial_cv,
            'spatial_skew':         spatial_skew,
            'spatial_iqr':          spatial_iqr,
            # concentration
            'top10_sum':            top10_sum,
            'top10_share':          top10_share,
            'gini_coeff':           gini_coeff,
        }

        # Add PCA factors
        for k, factor_val in enumerate(pca_factors):
            row[f'pca_factor_{k+1}'] = factor_val

        spatial_records.append(row)

    spatial_df = pd.DataFrame(spatial_records).set_index('horizon')
    print(f"  [INFO] Spatial features: {spatial_df.shape[0]} horizons x "
          f"{spatial_df.shape[1]} features")

    return spatial_df, pca, scaler_pca


# ==============================================================================
# STEP 6: WALK-FORWARD VALIDATION
# ==============================================================================

def walk_forward_validate_stage1(
    panel:      pd.DataFrame,   # (n_months, n_cities)
    horizon:    int  = 6,
    min_train:  int  = 36,      # minimum months before first validation
    step:       int  = 3,       # roll forward every N months
    val_months: int  = 12
) -> pd.DataFrame:
    """
    Walk-forward validation of Stage 1 residual chain.

    For each fold:
        train  -> all months up to t
        predict -> X_hat(t+1) ... X_hat(t+6)
        score  -> MAE per horizon vs actual panel values

    Returns:
        results_df: (n_folds * horizon, cols)
        columns: [fold, horizon, city, mae, rmse, mape]
    """
    n_months = len(panel)
    results  = []
    fold     = 0

    for t in range(min_train, n_months - horizon, step):
        fold += 1
        panel_train = panel.iloc[:t]

        print(f"\n  [FOLD {fold}] Training on {t} months -> predicting h=1..{horizon}")

        # Fit pipeline on training data only
        features_df, le, city_stats = build_monthly_features(panel_train, horizon)
        horizon_data  = build_residual_chain_targets(features_df, panel_train, horizon)
        models        = train_residual_chain_models(horizon_data, horizon, val_months)

        # Predict
        X_hat_monthly = predict_residual_chain(models, panel_train, le, city_stats, horizon)

        # Score against actual
        for h in range(1, horizon + 1):
            actual = panel.iloc[t + h - 1]   # actual panel values at t+h
            preds  = X_hat_monthly.loc[h]

            common_cities = actual.index.intersection(preds.index)
            a = actual[common_cities].values
            p = preds[common_cities].values

            mae  = mean_absolute_error(a, p)
            rmse = np.sqrt(mean_squared_error(a, p))
            mape = np.mean(np.abs((a - p) / (np.abs(a) + 1e-8))) * 100

            results.append({
                'fold':    fold,
                'horizon': h,
                'mae':     mae,
                'rmse':    rmse,
                'mape':    mape,
            })

    results_df = pd.DataFrame(results)
    summary    = results_df.groupby('horizon')[['mae', 'rmse', 'mape']].mean()
    print("\n  ===== STAGE 1 VALIDATION  -  MEAN METRICS BY HORIZON =====")
    print(summary.round(4).to_string())

    return results_df


# ==============================================================================
# MASTER STAGE 1 PIPELINE
# ==============================================================================

def run_stage1(
    df_transactions: pd.DataFrame,   # raw transaction data
    date_col:        str  = 'date',
    city_col:        str  = 'city',
    value_col:       str  = 'value',
    agg_func:        str  = 'sum',
    horizon:         int  = 6,
    n_pca_components:int  = 15,
    validate:        bool = True
) -> dict:
    """
    Full Stage 1 pipeline.

    Returns artifacts dict containing everything Stage 2 needs:
    {
        'X_hat_monthly':  (6, n_cities)           -  city-level forecasts
        'X_hat_spatial':  (6, n_spatial_features)  -  spatial features for Stage 2
        'panel':          (n_months, n_cities)     -  historical monthly panel
        'features_df':    long format features     -  for Stage 2 reference
        'models':         {h: LGBModel}            -  trained models
        'le':             LabelEncoder             -  city encoder
        'city_stats':     dict                     -  city-level statistics
        'pca':            PCA object               -  spatial PCA
        'scaler_pca':     StandardScaler           -  PCA scaler
        'val_results':    DataFrame                -  validation metrics (if validate=True)
    }
    """
    print("=" * 60)
    print("STAGE 1: GLOBAL LIGHTGBM WITH RESIDUAL CHAINING")
    print("=" * 60)

    print("\n[1/6] Aggregating transactions -> monthly panel...")
    panel = aggregate_transactions_to_monthly(
        df_transactions, date_col, city_col, value_col, agg_func
    )

    print("\n[2/6] Building feature matrix...")
    features_df, le, city_stats = build_monthly_features(panel, horizon)

    print("\n[3/6] Building residual chain targets...")
    horizon_data = build_residual_chain_targets(features_df, panel, horizon)

    print("\n[4/6] Training GlobalLightGBM residual chain models...")
    models = train_residual_chain_models(horizon_data, horizon)

    print("\n[5/6] Predicting forward 6 months...")
    X_hat_monthly = predict_residual_chain(models, panel, le, city_stats, horizon)

    print("\n[6/6] Engineering spatial features for Stage 2...")
    X_hat_spatial, pca, scaler_pca = engineer_spatial_features(
        X_hat_monthly, panel, n_pca_components
    )

    val_results = None
    if validate:
        print("\n[VAL] Running walk-forward validation...")
        val_results = walk_forward_validate_stage1(panel, horizon)

    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETE")
    print(f"  X_hat_monthly shape : {X_hat_monthly.shape}  (horizons x cities)")
    print(f"  X_hat_spatial shape : {X_hat_spatial.shape}  (horizons x spatial features)")
    print("=" * 60)

    return {
        'X_hat_monthly':  X_hat_monthly,
        'X_hat_spatial':  X_hat_spatial,
        'panel':          panel,
        'features_df':    features_df,
        'models':         models,
        'le':             le,
        'city_stats':     city_stats,
        'pca':            pca,
        'scaler_pca':     scaler_pca,
        'val_results':    val_results,
    }


# ==============================================================================
# QUICK SMOKE TEST WITH SYNTHETIC DATA
# ==============================================================================

if __name__ == '__main__':

    print("Generating synthetic transaction data...")
    np.random.seed(42)

    n_cities       = 20
    n_months       = 96
    n_transactions = 50000

    cities = [f'city_{i:03d}' for i in range(n_cities)]

    # Generate synthetic monthly panel with trend + seasonality + noise
    dates, city_labels, values = [], [], []
    start = pd.Timestamp('2016-01-01')

    for _ in range(n_transactions):
        city  = np.random.choice(cities)
        month = np.random.randint(0, n_months)
        date  = start + pd.DateOffset(months=month) + pd.DateOffset(days=np.random.randint(0, 28))

        # City-specific trend + shared seasonality + noise
        city_idx  = cities.index(city)
        trend     = 100 + city_idx * 5 + month * 0.3
        seasonal  = 20 * np.sin(2 * np.pi * date.month / 12)
        noise     = np.random.normal(0, 10)
        value     = max(0, trend + seasonal + noise)

        dates.append(date)
        city_labels.append(city)
        values.append(value)

    df_tx = pd.DataFrame({'date': dates, 'city': city_labels, 'value': values})
    print(f"Synthetic transactions: {len(df_tx):,} rows")

    # Run Stage 1
    artifacts = run_stage1(
        df_tx,
        date_col='date',
        city_col='city',
        value_col='value',
        agg_func='sum',
        horizon=6,
        n_pca_components=5,
        validate=False       # set True for full validation
    )

    print("\nX_hat_monthly (first 3 cities):")
    print(artifacts['X_hat_monthly'].iloc[:, :3].round(2))

    print("\nX_hat_spatial (spatial features for Stage 2):")
    print(artifacts['X_hat_spatial'].round(4))
