"""
数据获取与缓存模块。
支持 yfinance、富途、OpenBB 多数据源，统一缓存到 ./data/。
"""

import glob
import os
import time

import numpy as np
import pandas as pd

from .config import DATA_DIR, DEFAULT_PROVIDER, DEFAULT_START_DATE, FUTU_HOST, FUTU_PORT, IV_HORIZON_DAYS, MAX_RETRIES, RETRY_DELAY
from .iv_module import get_futu_iv


def _cache_path(symbol: str, start_date: str, end_date: str) -> str:
    """本地缓存路径：按 (symbol, start_date, end_date) 索引，end_date=None 记为 latest。"""
    end_str = end_date if end_date else "latest"
    safe = lambda s: s.replace("-", "") if s else s
    return os.path.join(DATA_DIR, f"{symbol}_{safe(start_date)}_{safe(end_str)}.csv")


def _symbol_to_yf(symbol: str) -> str:
    """转为 yfinance 格式。AAPL/US.AAPL->AAPL；9992.hk/HK.09992->9992.HK。"""
    s = (symbol or "").strip().upper()
    if not s:
        return symbol
    if s.startswith("HK."):
        c = s[3:].lstrip("0") or "0"
        return c.zfill(4) + ".HK"
    if s.startswith("US."):
        return s[3:]
    if s.endswith(".HK") or s.endswith(".SH") or s.endswith(".SZ"):
        return s
    return s


def _symbol_to_futu(symbol: str):
    """将 yfinance/OpenBB 格式转为富途格式。HK.00700/US.AAPL 已是富途格式；0700.HK->HK.00700；AAPL->US.AAPL。"""
    s = (symbol or "").strip().upper()
    if not s:
        return None
    if s.startswith("HK.") or s.startswith("US."):
        return s
    if s.endswith(".HK"):
        code = s[:-3]
        return "HK." + code.zfill(5)
    if s.endswith(".SH") or s.endswith(".SZ"):
        return None
    return "US." + s


def _get_yfinance_prices(symbol: str, start_date: str, end_date: str) -> pd.Series | None:
    """用 yfinance 获取日 K 收盘价，失败返回 None。"""
    import yfinance as yf

    try:
        ticker = yf.Ticker(_symbol_to_yf(symbol))
        end = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")
        df = ticker.history(start=start_date, end=end, auto_adjust=False)
        if df is None or df.empty:
            return None
        close = df["Close"] if "Close" in df.columns else df["close"]
        close = close.sort_index()
        close.index = close.index.normalize()
        return close
    except Exception:
        return None


def _get_futu_prices(futu_code: str, start_date: str, end_date: str, host: str = FUTU_HOST, port: int = FUTU_PORT, quote_ctx=None) -> pd.Series:
    """从富途 request_history_kline 获取日 K 收盘价序列，前复权。"""
    from futu import OpenQuoteContext, RET_OK, KLType, AuType

    own_ctx = quote_ctx is None
    if own_ctx:
        quote_ctx = OpenQuoteContext(host=host, port=port)
    try:
        all_rows = []
        page_req_key = None
        while True:
            ret, data, page_req_key = quote_ctx.request_history_kline(
                futu_code,
                start=start_date,
                end=end_date or pd.Timestamp.now().strftime("%Y-%m-%d"),
                ktype=KLType.K_DAY,
                autype=AuType.QFQ,
                max_count=1000,
                page_req_key=page_req_key,
            )
            if ret != RET_OK:
                raise ValueError(f"富途 K 线获取失败: {data}")
            if data is None or data.empty:
                break
            all_rows.append(data)
            if page_req_key is None:
                break
        if not all_rows:
            raise ValueError(f"富途未返回数据: {futu_code}")
        df = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["time_key"])
        df["time_key"] = pd.to_datetime(df["time_key"])
        df = df.sort_values("time_key")
        close = df.set_index("time_key")["close"].squeeze()
        close.index = close.index.normalize()
        return close
    finally:
        if own_ctx:
            quote_ctx.close()


def list_cached_data() -> pd.DataFrame:
    """列出 ./data/ 下按 (标的, 起始日期, 结束日期) 索引的价格缓存。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
        name = os.path.basename(path)
        if name.endswith("_iv.csv") or name.endswith("_iv_term.csv") or name.endswith("_iv_series.csv"):
            continue
        base = name[:-4]
        parts = base.split("_")
        if len(parts) != 3:
            continue
        sym, start_raw, end_raw = parts
        start_d = f"{start_raw[:4]}-{start_raw[4:6]}-{start_raw[6:8]}" if len(start_raw) == 8 else start_raw
        end_d = end_raw if end_raw == "latest" else (f"{end_raw[:4]}-{end_raw[4:6]}-{end_raw[6:8]}" if len(end_raw) == 8 else end_raw)
        rows.append({"symbol": sym, "start_date": start_d, "end_date": end_d, "path": path})
    return pd.DataFrame(rows)


def get_stock_prices(
    symbol: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = None,
    max_retries: int = MAX_RETRIES,
    retry_delay: int = RETRY_DELAY,
    provider: str = DEFAULT_PROVIDER,
    host: str = FUTU_HOST,
    port: int = FUTU_PORT,
    horizon: int = IV_HORIZON_DAYS,
    quote_ctx=None,
):
    """
    按 (symbol, start_date, end_date) 索引：先查 ./data/ 缓存，无则优先 yfinance，其次富途，最后 OpenBB。
    同时计算 HV、拉取 IV（富途标的用 Futu IV）。返回 (prices, hv_annual_pct, iv_annual_pct)。
    """
    path = _cache_path(symbol, start_date, end_date)
    futu_code = _symbol_to_futu(symbol)

    if os.path.isfile(path):
        s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
        close = s.sort_index()
    else:
        close = _get_yfinance_prices(symbol, start_date, end_date)
        if close is not None and len(close) > 0:
            os.makedirs(DATA_DIR, exist_ok=True)
            close.to_csv(path)
        elif futu_code:
            close = _get_futu_prices(futu_code, start_date, end_date, host, port, quote_ctx)
            os.makedirs(DATA_DIR, exist_ok=True)
            close.to_csv(path)
        else:
            from openbb import obb

            last_err = None
            df = None
            for attempt in range(max_retries):
                try:
                    result = obb.equity.price.historical(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        interval="1d",
                        provider=provider,
                    )
                    df = result.to_df()
                    break
                except Exception as e:
                    last_err = e
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        raise last_err
            if df is None or df.empty:
                raise ValueError(f"OpenBB 未返回数据: {symbol}")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
            close = df["close"] if "close" in df.columns else df["Close"]
            close = close.sort_index()
            os.makedirs(DATA_DIR, exist_ok=True)
            close.to_csv(path)

    ret = returns_from_prices(close)
    ret_horizon = ret.tail(horizon) if len(ret) >= horizon else ret
    hv_annual_pct = float(np.std(ret_horizon) * np.sqrt(252) * 100)

    if futu_code:
        r = get_futu_iv(futu_code, horizon=IV_HORIZON_DAYS, verbose=False, host=host, port=port, quote_ctx=quote_ctx, data_dir=DATA_DIR)
        iv_annual_pct = r["iv_annual_pct"] if r.get("error") is None else np.nan
    else:
        iv_annual_pct = np.nan

    return close, hv_annual_pct, iv_annual_pct


def returns_from_prices(prices: pd.Series) -> pd.Series:
    """从价格序列计算对数收益率。"""
    return np.log(prices / prices.shift(1)).dropna()
