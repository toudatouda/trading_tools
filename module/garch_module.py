"""
GARCH 波动率拟合与预测模块。
"""

import numpy as np
import pandas as pd
from arch import arch_model

from .config import GARCH_HORIZON_DAYS, HOLD_PERIOD, IV_HORIZON_DAYS
from .data_module import get_stock_prices, returns_from_prices


def get_volatility_metrics(
    symbol: str,
    start_date: str = "2010-01-01",
    end_date: str = None,
    garch_horizon_days: int = GARCH_HORIZON_DAYS,
    iv_horizon_days: int = IV_HORIZON_DAYS,
    hold_period: int = HOLD_PERIOD,
    quote_ctx=None,
):
    """
    一站式：价格+HV+IV（get_stock_prices 统一富途/OpenBB 并缓存）→ GARCH 拟合与预测。
    返回 dict：prices, returns, fit, forecast, hv_annual_pct, iv_annual_pct,
    garch_1d_pct, garch_nd_pct, vol_table, cond_vol_pct, annual_vol_pct,
    cond_vol_hist, garch_vol_series, hv_series, horizon_days。
    """
    prices, hv_annual_pct, iv_annual_pct = get_stock_prices(symbol, start_date, end_date,garch_horizon_days=garch_horizon_days, iv_horizon_days=iv_horizon_days, quote_ctx=quote_ctx)
    returns = returns_from_prices(prices)
    roll_window = garch_horizon_days
    hv_series = returns.rolling(roll_window).std() * np.sqrt(252) * 100
    hv_series = hv_series.dropna()

    iv_daily_pct = (iv_annual_pct / 100) / np.sqrt(252) if not np.isnan(iv_annual_pct) else np.nan
    last_price = prices.iloc[-1] if hasattr(prices, "iloc") else prices[-1]

    model = arch_model(returns * 100, mean="Constant", vol="GARCH", p=1, q=1, rescale=False)
    fit = model.fit(disp="off")
    forecast = fit.forecast(horizon=garch_horizon_days, reindex=False)
    cond_var = forecast.variance.values[-1, :]
    cond_vol_pct = np.sqrt(cond_var)
    annual_vol_pct = cond_vol_pct * np.sqrt(252)
    garch_1d_pct = float(annual_vol_pct[0])
    garch_nd_pct = float(annual_vol_pct[-1])
    expected_move = (garch_nd_pct / np.sqrt(252)) * 0.01 * np.sqrt(garch_horizon_days * hold_period) * last_price
    lower_bound = round(last_price - expected_move, 2)
    upper_bound = round(last_price + expected_move, 2)

    vol_table = pd.DataFrame({
        "HV annual %": [round(hv_annual_pct, 2)],
        "IV annual %": [round(iv_annual_pct, 2) if not np.isnan(iv_annual_pct) else "—"],
        "GARCH T+1 annual %": [round(garch_1d_pct, 2)],
        f"GARCH T+{garch_horizon_days} annual %": [round(garch_nd_pct, 2)],
        "price": [round(last_price, 2)],
        "expected move": [round(expected_move, 2)],
        "expected range": [f"{lower_bound} ~ {upper_bound}"],
    })
    vol_table.index = [symbol]

    garch_vol_series = fit.conditional_volatility * np.sqrt(252)

    return {
        "prices": prices,
        "returns": returns,
        "fit": fit,
        "forecast": forecast,
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
