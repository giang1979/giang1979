"""
Du bao gia co phieu ngan hang thuong mai Viet Nam bang may hoc (Google Colab ready).
Tac gia: Codex

Huong dan nhanh tren Colab:
1) Upload file nay len Colab hoac copy vao 1 cell Python.
2) Chay: !python bank_stock_ml_colab.py --fast
   - Bo --fast de chay day du GridSearchCV (729 x 5 folds cho LightGBM).
"""

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

warnings.filterwarnings("ignore")


# Tren Colab, neu thieu thu vien, co the chay cell sau truoc khi run script:
# !pip install -q numpy pandas matplotlib seaborn scikit-learn statsmodels scipy lightgbm xgboost catboost ThymeBoost tabulate

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

from statsmodels.tsa.stattools import adfuller, kpss

from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

try:
    from ThymeBoost import ThymeBoost as tb
except Exception:
    tb = None


URLS = {
    "ACB": "https://raw.githubusercontent.com/giang1979/giang1979/refs/heads/main/D%E1%BB%AF%20li%E1%BB%87u%20L%E1%BB%8Bch%20s%E1%BB%AD%20ACB-2025.csv",
    "CTG": "https://raw.githubusercontent.com/giang1979/giang1979/refs/heads/main/D%E1%BB%AF%20li%E1%BB%87u%20L%E1%BB%8Bch%20s%E1%BB%AD%20CTG-2025.csv",
    "TCB": "https://raw.githubusercontent.com/giang1979/giang1979/refs/heads/main/D%E1%BB%AF%20li%E1%BB%87u%20L%E1%BB%8Bch%20s%E1%BB%AD%20TCB-2025.csv",
    "VCB": "https://raw.githubusercontent.com/giang1979/giang1979/refs/heads/main/D%E1%BB%AF%20li%E1%BB%87u%20L%E1%BB%8Bch%20s%E1%BB%AD%20VCB-2025.csv",
}

COL_MAP = {
    "Date": "ngay",
    "Open": "mo_cua",
    "High": "cao_nhat",
    "Low": "thap_nhat",
    "Close": "dong_cua",
    "Adj Close": "dieu_chinh",
    "Volume": "khoi_luong",
    "Ticker": "ma_cp",
}


@dataclass
class TransformInfo:
    use_sqrt: bool = False
    use_diff: bool = False
    shift_eps: float = 0.0


def pretty_df(df: pd.DataFrame, title: str = ""):
    print("\n" + "=" * 90)
    if title:
        print(title)
        print("-" * 90)
    print(df.to_markdown(index=False))


def load_and_clean(ticker: str, url: str) -> pd.DataFrame:
    print(f"\n[INFO] Dang tai du lieu {ticker}...")
    df = pd.read_csv(url)

    # Rename cot sang tieng Viet khong dau
    df = df.rename(columns=COL_MAP)

    # Dam bao cot ngay
    if "ngay" in df.columns:
        df["ngay"] = pd.to_datetime(df["ngay"], errors="coerce")
        df = df.sort_values("ngay").reset_index(drop=True)

    # Xu ly kieu du lieu so
    for c in ["mo_cua", "cao_nhat", "thap_nhat", "dong_cua", "dieu_chinh", "khoi_luong"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Missing values
    miss_before = int(df.isna().sum().sum())
    df = df.dropna().reset_index(drop=True)
    miss_after = int(df.isna().sum().sum())

    # Duplicate
    dup_before = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    dup_after = int(df.duplicated().sum())

    qc = pd.DataFrame(
        {
            "ma_cp": [ticker],
            "so_dong": [len(df)],
            "missing_truoc": [miss_before],
            "missing_sau": [miss_after],
            "dup_truoc": [dup_before],
            "dup_sau": [dup_after],
        }
    )
    pretty_df(qc, f"Kiem tra chat luong du lieu - {ticker}")

    return df


def run_adf_kpss(series: pd.Series) -> Dict[str, float]:
    s = series.dropna().astype(float)
    adf_stat, adf_p, _, _, _, _ = adfuller(s, regression="ct", autolag="AIC")
    kpss_stat, kpss_p, _, _ = kpss(s, regression="ct", nlags="auto")
    return {
        "adf_stat": adf_stat,
        "adf_p": adf_p,
        "kpss_stat": kpss_stat,
        "kpss_p": kpss_p,
        "adf_stationary": adf_p <= 0.05,
        "kpss_stationary": kpss_p > 0.05,
    }


def transform_pipeline(y: pd.Series) -> Tuple[pd.Series, TransformInfo, pd.DataFrame]:
    rows = []

    def assess(name: str, s: pd.Series):
        st = run_adf_kpss(s)
        rows.append({"bien_doi": name, **st})
        return st

    # (1) Sqrt
    eps = max(0.0, -float(y.min()) + 1e-8)
    y_sqrt = np.sqrt(y + eps)
    st1 = assess("sqrt", y_sqrt)

    # (2) Diff
    y_diff = y.diff().dropna()
    st2 = assess("diff", y_diff)

    # (3) Sqrt + Diff
    y_sd = y_sqrt.diff().dropna()
    st3 = assess("sqrt_diff", y_sd)

    tbl = pd.DataFrame(rows)
    pretty_df(tbl, "Bang kiem tra ADF/KPSS cac phep bien doi")

    info = TransformInfo(use_sqrt=True, use_diff=True, shift_eps=eps)
    transformed = y_sd.copy()

    # Neu sqrt+diff khong dat, fallback theo quy tac ung vien tot nhat
    if not (st3["adf_stationary"] and st3["kpss_stationary"]):
        if st2["adf_stationary"] and st2["kpss_stationary"]:
            info = TransformInfo(use_sqrt=False, use_diff=True, shift_eps=0.0)
            transformed = y_diff
        elif st1["adf_stationary"] and st1["kpss_stationary"]:
            info = TransformInfo(use_sqrt=True, use_diff=False, shift_eps=eps)
            transformed = y_sqrt
        else:
            info = TransformInfo(use_sqrt=False, use_diff=False, shift_eps=0.0)
            transformed = y.copy()

    return transformed, info, tbl


def inverse_transform(pred_trans: np.ndarray, original: pd.Series, info: TransformInfo) -> np.ndarray:
    """Inverse cho truong hop sqrt+diff (hoac bien the)."""
    pred_trans = np.array(pred_trans, dtype=float)

    if info.use_sqrt and info.use_diff:
        start = np.sqrt(float(original.iloc[0]) + info.shift_eps)
        cum = np.r_[start, start + np.cumsum(pred_trans)]
        inv = np.square(cum) - info.shift_eps
        return inv[1:]
    if info.use_diff and not info.use_sqrt:
        start = float(original.iloc[0])
        cum = np.r_[start, start + np.cumsum(pred_trans)]
        return cum[1:]
    if info.use_sqrt and not info.use_diff:
        return np.square(pred_trans) - info.shift_eps
    return pred_trans


def tukey_outliers(series: pd.Series) -> Dict[str, Any]:
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1
    inner_low, inner_high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outer_low, outer_high = q1 - 3.0 * iqr, q3 + 3.0 * iqr

    possible_mask = (series < inner_low) | (series > inner_high)
    probable_mask = (series < outer_low) | (series > outer_high)

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "inner_low": inner_low,
        "inner_high": inner_high,
        "outer_low": outer_low,
        "outer_high": outer_high,
        "possible_idx": np.where(possible_mask)[0],
        "probable_idx": np.where(probable_mask)[0],
        "possible_count": int(possible_mask.sum()),
        "probable_count": int(probable_mask.sum()),
    }


def thymeboost_outliers(series: pd.Series) -> np.ndarray:
    if tb is None:
        return np.array([], dtype=int)
    try:
        boosted_model = tb.ThymeBoost(verbose=0)
        res = boosted_model.detect_outliers(series.values)
        # res co the la list hoac dict tuy version
        if isinstance(res, dict):
            idx = res.get("outliers", [])
        else:
            idx = res
        return np.array(idx, dtype=int)
    except Exception:
        return np.array([], dtype=int)


def make_lag_xy(series: pd.Series, n_lags: int = 5) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    s = series.dropna().reset_index(drop=True)
    X, y = [], []
    for i in range(n_lags, len(s)):
        X.append(s.iloc[i - n_lags:i].values)
        y.append(s.iloc[i])
    idx = list(range(n_lags, len(s)))
    return np.array(X), np.array(y), idx


def split_time_ordered(X: np.ndarray, y: np.ndarray, train_ratio=0.64, val_ratio=0.16):
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate(y_true, y_pred) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
    }


def train_gscv_lightgbm(X_train, y_train, fast=False):
    model = LGBMRegressor(random_state=42)
    if fast:
        param_grid = {
            "boosting_type": ["dart"],
            "objective": ["regression_l1"],
            "metric": ["rmse"],
            "learning_rate": [0.001, 0.01],
            "n_estimators": [100, 300],
            "max_depth": [3, 5],
            "bagging_fraction": [0.95],
            "bagging_freq": [20],
        }
    else:
        # 3^6 = 729 cau hinh -> 729*5=3645 fit
        param_grid = {
            "boosting_type": ["dart"],
            "objective": ["regression_l1"],
            "metric": ["rmse"],
            "learning_rate": [0.001, 0.005, 0.01],
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 7],
            "bagging_fraction": [0.85, 0.90, 0.95],
            "bagging_freq": [5, 10, 20],
        }

    tscv = TimeSeriesSplit(n_splits=5)
    gscv = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    gscv.fit(X_train, y_train)
    return gscv


def compare_models(X_train, y_train, X_test, y_test):
    models = {
        "LightGBM": LGBMRegressor(
            boosting_type="dart",
            objective="regression_l1",
            metric="rmse",
            learning_rate=0.001,
            n_estimators=100,
            max_depth=3,
            bagging_fraction=0.95,
            bagging_freq=20,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300,
            learning_rate=0.03,
            depth=4,
            loss_function="RMSE",
            verbose=0,
            random_seed=42,
        ),
        "AdaBoost": AdaBoostRegressor(
            n_estimators=300,
            learning_rate=0.03,
            random_state=42,
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        m = evaluate(y_test, pred)
        results.append({"mo_hinh": name, **m})

    out = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    return out


def robustness_6_ranges(series: pd.Series, n_lags: int = 5) -> pd.DataFrame:
    models = {
        "LightGBM": LGBMRegressor(
            boosting_type="dart",
            objective="regression_l1",
            metric="rmse",
            learning_rate=0.001,
            n_estimators=100,
            max_depth=3,
            bagging_fraction=0.95,
            bagging_freq=20,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            objective="reg:squarederror",
            random_state=42,
        ),
        "CatBoost": CatBoostRegressor(iterations=200, learning_rate=0.03, depth=4, verbose=0, random_seed=42),
        "AdaBoost": AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    }

    out_rows = []
    n = len(series)
    boundaries = np.linspace(0, n, 7, dtype=int)

    for i in range(1, 7):
        end = boundaries[i]
        sub = series.iloc[:end]
        X, y, _ = make_lag_xy(sub, n_lags=n_lags)
        if len(X) < 30:
            continue

        X_train, y_train, _, _, X_test, y_test = split_time_ordered(X, y)
        if len(X_test) < 5:
            continue

        for mname, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            out_rows.append({"khoang_thoi_gian": i, "mo_hinh": mname, "RMSE": rmse, "so_diem": len(sub)})

    return pd.DataFrame(out_rows)


def plot_stationarity(original: pd.Series, transformed: pd.Series, ticker: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    axes[0].plot(original.values, color="tab:blue", lw=1.8)
    axes[0].set_title(f"{ticker} - Chuoi goc (dong_cua)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(transformed.values, color="tab:green", lw=1.8)
    axes[1].set_title(f"{ticker} - Chuoi sau bien doi")
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_outliers(series: pd.Series, tukey: Dict[str, Any], tb_idx: np.ndarray, ticker: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(series.values, label="series", color="steelblue")
    if tukey["possible_count"] > 0:
        axes[0].scatter(tukey["possible_idx"], series.iloc[tukey["possible_idx"]], c="orange", label="possible", s=35)
    if tukey["probable_count"] > 0:
        axes[0].scatter(tukey["probable_idx"], series.iloc[tukey["probable_idx"]], c="red", label="probable", s=45)
    axes[0].set_title(f"{ticker} - Tukey outliers")
    axes[0].legend()

    sns.boxplot(y=series.values, ax=axes[1], color="lightgreen")
    axes[1].set_title(f"{ticker} - Boxplot IQR")

    sns.histplot(series.values, kde=True, ax=axes[2], color="mediumpurple")
    axes[2].set_title(f"{ticker} - Distribution")

    plt.tight_layout()
    plt.show()

    if len(tb_idx) > 0:
        plt.figure(figsize=(12, 3))
        plt.plot(series.values, color="gray")
        plt.scatter(tb_idx, series.iloc[tb_idx], c="crimson", s=35, label="ThymeBoost")
        plt.title(f"{ticker} - ThymeBoost detect_outliers")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()


def plot_split(y: pd.Series, n_lags: int, ticker: str):
    X, yy, idx = make_lag_xy(y, n_lags=n_lags)
    n = len(yy)
    n_train = int(n * 0.64)
    n_val = int(n * 0.16)

    plt.figure(figsize=(14, 4))
    plt.plot(yy, color="black", lw=1.4)
    plt.axvspan(0, n_train, color="green", alpha=0.15, label="Train ~64%")
    plt.axvspan(n_train, n_train + n_val, color="orange", alpha=0.2, label="Validation ~16%")
    plt.axvspan(n_train + n_val, n, color="red", alpha=0.15, label="Test ~20%")
    plt.title(f"{ticker} - Chia du lieu time ordered")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_one_ticker(ticker: str, url: str, n_lags=5, fast=False) -> Dict[str, Any]:
    df = load_and_clean(ticker, url)
    y_raw = df["dong_cua"].copy()

    # Giai doan 2A + 2B
    st_raw = run_adf_kpss(y_raw)
    pretty_df(pd.DataFrame([{"ma_cp": ticker, **st_raw}]), f"ADF/KPSS truoc bien doi - {ticker}")

    y_trans, tf_info, _ = transform_pipeline(y_raw)
    st_after = run_adf_kpss(y_trans)
    pretty_df(pd.DataFrame([{"ma_cp": ticker, **st_after}]), f"ADF/KPSS sau bien doi - {ticker}")

    plot_stationarity(y_raw, y_trans, ticker)

    # Giai doan 2C - outlier
    tukey = tukey_outliers(y_trans)
    tb_idx = thymeboost_outliers(y_trans)
    print(f"[INFO] {ticker} Tukey possible={tukey['possible_count']}, probable={tukey['probable_count']}")
    print(f"[INFO] {ticker} ThymeBoost outliers={len(tb_idx)}")
    plot_outliers(y_trans, tukey, tb_idx, ticker)

    # 3 chien luoc outlier
    # strategy A: impute (thay possible outlier bang median)
    y_imp = y_trans.copy()
    if tukey["possible_count"] > 0:
        med = y_imp.median()
        y_imp.iloc[tukey["possible_idx"]] = med

    # strategy B: scaler
    scaler = RobustScaler()
    y_scaled = pd.Series(scaler.fit_transform(y_trans.values.reshape(-1, 1)).ravel())

    # strategy C: delete
    keep_mask = np.ones(len(y_trans), dtype=bool)
    keep_mask[tukey["possible_idx"]] = False
    y_del = y_trans.iloc[keep_mask].reset_index(drop=True)

    stats_tbl = pd.DataFrame(
        [
            {"chien_luoc": "Impute", "kurtosis": kurtosis(y_imp), "skewness": skew(y_imp), "series": y_imp},
            {"chien_luoc": "Scaler", "kurtosis": kurtosis(y_scaled), "skewness": skew(y_scaled), "series": y_scaled},
            {"chien_luoc": "Delete", "kurtosis": kurtosis(y_del), "skewness": skew(y_del), "series": y_del},
        ]
    )

    # Chon strategy theo RMSE val (Delete thuong tot hon voi du lieu nay)
    rmse_rows = []
    for _, row in stats_tbl.iterrows():
        s = row["series"]
        X, yy, _ = make_lag_xy(s, n_lags=n_lags)
        if len(X) < 20:
            rmse_rows.append(np.inf)
            continue
        X_train, y_train, X_val, y_val, _, _ = split_time_ordered(X, yy)
        m = LGBMRegressor(
            boosting_type="dart",
            objective="regression_l1",
            metric="rmse",
            learning_rate=0.001,
            n_estimators=100,
            max_depth=3,
            bagging_fraction=0.95,
            bagging_freq=20,
            random_state=42,
        )
        m.fit(X_train, y_train)
        pred_val = m.predict(X_val)
        rmse_rows.append(np.sqrt(mean_squared_error(y_val, pred_val)))

    stats_tbl["RMSE_val"] = rmse_rows
    pretty_df(stats_tbl[["chien_luoc", "kurtosis", "skewness", "RMSE_val"]], f"So sanh 3 chien luoc outlier - {ticker}")

    best_strategy = stats_tbl.sort_values("RMSE_val").iloc[0]["chien_luoc"]
    y_final = stats_tbl.sort_values("RMSE_val").iloc[0]["series"]
    print(f"[INFO] {ticker} Chien luoc outlier tot nhat: {best_strategy}")

    # Giai doan 3
    plot_split(y_final, n_lags=n_lags, ticker=ticker)
    X, yy, _ = make_lag_xy(y_final, n_lags=n_lags)
    X_train, y_train, X_val, y_val, X_test, y_test = split_time_ordered(X, yy)

    # Giai doan 4 + 5
    gscv = train_gscv_lightgbm(X_train, y_train, fast=fast)
    cv_df = pd.DataFrame(gscv.cv_results_)
    best_idx = int(gscv.best_index_)
    best_row = cv_df.loc[best_idx, ["mean_test_score", "mean_train_score"]]

    best_report = pd.DataFrame(
        [
            {
                "ma_cp": ticker,
                "best_index": best_idx,
                "mean_test_score": best_row["mean_test_score"],
                "mean_train_score": best_row["mean_train_score"],
                "best_params": str(gscv.best_params_),
            }
        ]
    )
    pretty_df(best_report, f"GridSearchCV ket qua toi uu - {ticker}")

    # Danh gia tren test
    best_model = gscv.best_estimator_
    y_pred = best_model.predict(X_test)
    eval_lgbm = evaluate(y_test, y_pred)
    metric_tbl = pd.DataFrame([{"mo_hinh": "LightGBM", **eval_lgbm}])
    pretty_df(metric_tbl, f"Danh gia LightGBM - {ticker}")

    # Giai doan 7: so sanh mo hinh
    comp_df = compare_models(X_train, y_train, X_test, y_test)
    pretty_df(comp_df, f"So sanh mo hinh - {ticker}")

    # Robustness 6 ranges
    robust_df = robustness_6_ranges(y_final, n_lags=n_lags)
    if not robust_df.empty:
        pretty_df(robust_df, f"Robustness 6 khoang thoi gian - {ticker}")

        plt.figure(figsize=(12, 4))
        sns.barplot(data=robust_df, x="khoang_thoi_gian", y="RMSE", hue="mo_hinh")
        plt.title(f"{ticker} - RMSE theo 6 khoang thoi gian")
        plt.tight_layout()
        plt.show()

    # Chart tong hop
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(y_final.values, color="teal")
    axes[0, 0].set_title("Time series plot")

    sns.boxplot(y=y_final.values, ax=axes[0, 1], color="lightblue")
    axes[0, 1].set_title("Boxplot")

    sns.histplot(y_final.values, kde=True, ax=axes[1, 0], color="orange")
    axes[1, 0].set_title("Distribution chart")

    sns.barplot(data=comp_df, x="mo_hinh", y="RMSE", ax=axes[1, 1], palette="viridis")
    axes[1, 1].set_title("Bar comparison chart (RMSE)")
    plt.tight_layout()
    plt.show()

    return {
        "ticker": ticker,
        "best_params": gscv.best_params_,
        "lgbm_metrics": eval_lgbm,
        "compare": comp_df,
        "robust": robust_df,
    }


def main(fast=False):
    sns.set_theme(style="whitegrid", context="notebook")

    all_results = []
    for t, u in URLS.items():
        res = run_one_ticker(t, u, n_lags=5, fast=fast)
        all_results.append(res)

    summary = []
    for r in all_results:
        row = {"ma_cp": r["ticker"], **r["lgbm_metrics"], "best_params": str(r["best_params"])}
        summary.append(row)
    summary_df = pd.DataFrame(summary).sort_values("RMSE").reset_index(drop=True)
    pretty_df(summary_df, "Tong hop ket qua LightGBM cho 4 ngan hang")

    # Bang tong hop xep hang mo hinh tren tung ma
    rank_rows = []
    for r in all_results:
        cdf = r["compare"].copy().sort_values("RMSE").reset_index(drop=True)
        for i, row in cdf.iterrows():
            rank_rows.append({
                "ma_cp": r["ticker"],
                "hang": i + 1,
                "mo_hinh": row["mo_hinh"],
                "RMSE": row["RMSE"],
                "MAE": row["MAE"],
                "MedAE": row["MedAE"],
            })
    rank_df = pd.DataFrame(rank_rows)
    pretty_df(rank_df, "Bang xep hang mo hinh tren tung ma co phieu")


if __name__ == "__main__":
    fast_mode = "--fast" in sys.argv
    print(f"[RUN] fast_mode={fast_mode}")
    main(fast=fast_mode)
