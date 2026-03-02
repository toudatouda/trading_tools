"""
GARCH 波动率拟合与预测模块。
"""

import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from arch import arch_model

from .config import GARCH_HORIZON_DAYS, HOLD_PERIOD, IV_HORIZON_DAYS
from .data_module import get_stock_prices, returns_from_prices

MIN_SAMPLE_DAYS = 252  # GARCH 至少需要 1 年交易日

logger = logging.getLogger(__name__)

VOL_MODELS = ("GJR", "GARCH", "EGARCH")
DISTRIBUTIONS = ("skewt", "t", "normal")


def _fit_best_model(returns_pct: np.ndarray, vol_model: str, dist: str):
    """按指定 vol_model 和 dist 拟合，成功返回 (fit, model_spec)，失败返回 None。"""
    try:
        if vol_model == "GJR":
            model = arch_model(
                returns_pct, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist=dist, rescale=True
            )
        elif vol_model == "EGARCH":
            model = arch_model(
                returns_pct, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist=dist, rescale=True
            )
        else:
            model = arch_model(
                returns_pct, mean="Constant", vol="GARCH", p=1, q=1, dist=dist, rescale=True
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="arch")
            fit = model.fit(disp="off", options={"maxiter": 500})
        if fit.convergence_flag == 0:
            return fit, f"{vol_model}+{dist}"
    except Exception:
        pass
    return None


def get_volatility_metrics(
    symbol: str,
    start_date: str = "2010-01-01",
    end_date: str = None,
    garch_horizon_days: int = GARCH_HORIZON_DAYS,
    iv_horizon_days: int = IV_HORIZON_DAYS,
    hold_period: int = HOLD_PERIOD,
    quote_ctx=None,
    vol_model: Literal["auto", "GARCH", "GJR", "EGARCH"] = "auto",
    dist: Literal["auto", "normal", "t", "skewt"] = "auto",
):
    """
    一站式：价格+HV+IV（get_stock_prices 统一富途/OpenBB 并缓存）→ GARCH 拟合与预测。
    返回 dict：prices, returns, fit, forecast, hv_annual_pct, iv_annual_pct,
    garch_1d_pct, garch_nd_pct, vol_table, cond_vol_pct, annual_vol_pct,
    cond_vol_hist, garch_vol_series, hv_series, horizon_days。
    """
    prices, hv_annual_pct, iv_annual_pct = get_stock_prices(
        symbol, start_date, end_date, horizon=iv_horizon_days, quote_ctx=quote_ctx
    )
    returns = returns_from_prices(prices)

    if len(returns) < MIN_SAMPLE_DAYS:
        raise ValueError(
            f"GARCH 需要至少 {MIN_SAMPLE_DAYS} 个交易日，当前仅 {len(returns)} 个。"
            f"请将 start_date 提前（如 2020-01-01）以获取更多历史数据。"
        )

    roll_window = garch_horizon_days
    hv_series = returns.rolling(roll_window).std() * np.sqrt(252) * 100
    hv_series = hv_series.dropna()

    iv_daily_pct = (iv_annual_pct / 100) / np.sqrt(252) if not np.isnan(iv_annual_pct) else np.nan
    last_price = prices.iloc[-1] if hasattr(prices, "iloc") else prices[-1]

    returns_pct = (returns * 100).values
    model_spec = None

    if vol_model == "auto" or dist == "auto":
        best_aic, fit, model_spec = np.inf, None, None
        for vm in VOL_MODELS if vol_model == "auto" else [vol_model]:
            for d in DISTRIBUTIONS if dist == "auto" else [dist]:
                res = _fit_best_model(returns_pct, vm, d)
                if res is not None and res[0].aic < best_aic:
                    best_aic, fit, model_spec = res[0].aic, res[0], res[1]
        if fit is None:
            raise RuntimeError("GARCH 自动选模均未收敛，请扩大样本量或指定 vol_model/dist。")
    else:
        res = _fit_best_model(returns_pct, vol_model, dist)
        if res is None:
            raise RuntimeError(f"GARCH({vol_model}, {dist}) 拟合失败。")
        fit, model_spec = res

    # EGARCH/GJR 在 horizon>1 时不支持 analytic，需用 simulation
    fcast_method = "simulation" if garch_horizon_days > 1 else "analytic"
    forecast = fit.forecast(horizon=garch_horizon_days, method=fcast_method, reindex=False)
    scale = getattr(fit, "scale", 1.0)
    cond_var_raw = forecast.variance.values[-1, :]
    cond_var = cond_var_raw / (scale**2) if scale != 1.0 else cond_var_raw
    cond_vol_pct = np.sqrt(cond_var)
    annual_vol_pct = cond_vol_pct * np.sqrt(252)
    garch_1d_pct = float(annual_vol_pct[0])
    garch_nd_pct = float(annual_vol_pct[-1])

    n_hold = min(max(1, int(garch_horizon_days * hold_period)), garch_horizon_days)
    n_day_var = np.sum(cond_var[:n_hold])
    expected_move = last_price * 0.01 * np.sqrt(n_day_var)
    lower_bound = round(last_price - expected_move, 2)
    upper_bound = round(last_price + expected_move, 2)

    vol_table = pd.DataFrame({
        "model": [model_spec or "GARCH+normal"],
        "HV annual %": [round(hv_annual_pct, 2)],
        "IV annual %": [round(iv_annual_pct, 2) if not np.isnan(iv_annual_pct) else "—"],
        "GARCH T+1 annual %": [round(garch_1d_pct, 2)],
        f"GARCH T+{garch_horizon_days} annual %": [round(garch_nd_pct, 2)],
        "price": [round(last_price, 2)],
        "expected move": [round(expected_move, 2)],
        "expected range": [f"{lower_bound} ~ {upper_bound}"],
    })
    vol_table.index = [symbol]

    cond_vol_hist_scaled = fit.conditional_volatility / scale if scale != 1.0 else fit.conditional_volatility
    garch_vol_series = pd.Series(cond_vol_hist_scaled * np.sqrt(252), index=returns.index)

    return {
        "prices": prices,
        "returns": returns,
        "fit": fit,
        "forecast": forecast,
        "model_spec": model_spec,
        "hv_annual_pct": hv_annual_pct,
        "iv_annual_pct": iv_annual_pct,
        "garch_1d_pct": garch_1d_pct,
        "garch_nd_pct": garch_nd_pct,
        "vol_table": vol_table,
        "cond_vol_pct": cond_vol_pct,
        "annual_vol_pct": annual_vol_pct,
        "cond_vol_hist": fit.conditional_volatility,
        "garch_vol_series": garch_vol_series,
        "hv_series": hv_series,
        "horizon_days": garch_horizon_days,
        "iv_horizon_days": iv_horizon_days
    }
