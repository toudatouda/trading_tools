"""
隐含波动率（IV）计算模块。
通过富途 OpenD 获取期权数据，支持 VIX 风格与方差空间插值两种方法。
"""

import json
import math
import os
import time
from datetime import date

import numpy as np

from .config import DATA_DIR, FUTU_HOST, FUTU_PORT, IV_DTE_RANGE, IV_HORIZON_DAYS, IV_METHOD, RISK_FREE_RATE


def _compute_variance_vix_style(df_chain, df_snap, spot, dte, r=0.04):
    """
    VIX 风格单期限方差：方差互换公式，使用 OTM 期权价格。
    σ² = (2/T)Σ(ΔK/K²)e^(rT)Q(K) - (1/T)[F/K₀-1]²
    """
    T = dte / 365.0
    if T <= 0:
        return None
    erT = math.exp(r * T)

    price_cols = ["code"]
    for c in ["bid_price", "ask_price", "last_price"]:
        if c in df_snap.columns:
            price_cols.append(c)
    if len(price_cols) < 2:
        return None
    df = df_chain.merge(df_snap[price_cols], on="code", how="inner")
    df["option_type"] = df["option_type"].astype(str).str.upper()
    df["strike_price"] = df["strike_price"].astype(float)

    def _q(row):
        b = row.get("bid_price") if "bid_price" in row.index else 0
        a = row.get("ask_price") if "ask_price" in row.index else 0
        b, a = float(b or 0), float(a or 0)
        if b > 0 and a > 0:
            return (b + a) / 2
        return float(row.get("last_price", 0) or 0)

    df["Q"] = df.apply(_q, axis=1)
    df = df[df["Q"] > 0].copy()
    if df.empty:
        return None

    puts = df[df["option_type"].str.contains("PUT", na=False)].set_index("strike_price")["Q"].to_dict()
    calls = df[df["option_type"].str.contains("CALL", na=False)].set_index("strike_price")["Q"].to_dict()

    strikes = sorted(set(puts.keys()) | set(calls.keys()))
    if len(strikes) < 3:
        return None

    min_diff, k_atm = float("inf"), strikes[len(strikes) // 2]
    for k in strikes:
        c = calls.get(k, 0)
        p = puts.get(k, 0)
        if c > 0 and p > 0:
            diff = abs(c - p)
            if diff < min_diff:
                min_diff, k_atm = diff, k
    F = k_atm + erT * (calls.get(k_atm, 0) - puts.get(k_atm, 0))
    if F <= 0:
        F = spot * erT if spot else k_atm

    k0_candidates = [k for k in strikes if k <= F]
    K0 = max(k0_candidates) if k0_candidates else min(strikes)

    contribs = []
    for k in strikes:
        if k < K0:
            q = puts.get(k, 0)
        elif k > K0:
            q = calls.get(k, 0)
        else:
            q = (puts.get(k, 0) + calls.get(k, 0)) / 2 if (puts.get(k, 0) and calls.get(k, 0)) else puts.get(k, 0) or calls.get(k, 0)
        if q <= 0:
            continue
        contribs.append((k, q))

    if len(contribs) < 2:
        return None

    contribs.sort(key=lambda x: x[0])
    k_list = [x[0] for x in contribs]

    def delta_k(i):
        if i == 0:
            return k_list[1] - k_list[0]
        if i == len(k_list) - 1:
            return k_list[-1] - k_list[-2]
        return (k_list[i + 1] - k_list[i - 1]) / 2

    sigma_term = 0.0
    for i, (k, q) in enumerate(contribs):
        dk = delta_k(i)
        sigma_term += (dk / (k * k)) * erT * q

    correction = (1 / T) * ((F / K0) - 1) ** 2
    sigma2 = (2 / T) * sigma_term - correction
    return max(0.0, sigma2)


def get_futu_iv(
    underlying: str,
    horizon: int = IV_HORIZON_DAYS,
    *,
    host: str = FUTU_HOST,
    port: int = FUTU_PORT,
    dte_range: tuple = IV_DTE_RANGE,
    method: str = IV_METHOD,
    risk_free_rate: float = RISK_FREE_RATE,
    quote_ctx=None,
    verbose: bool = True,
    data_dir: str = DATA_DIR,
) -> dict:
    """
    通过富途 OpenD 获取标的未来 horizon 日整体隐含波动率（年化 %）。

    method="vix_style": VIX 风格，全行权价 OTM 期权 + 方差互换公式。
    method="variance_interp": 简化版，ATM IV + 方差空间插值。

    Returns
    -------
    dict
        - iv_annual_pct: 插值整体 IV（年化 %）
        - spot: 标的现价
        - iv_by_dte: [(dte, iv), ...]
        - sigma2_by_dte: [(dte, sigma2), ...]（vix_style 时）
        - interp_dtes, interp_weights, method, error
    """
    from futu import OpenQuoteContext, RET_OK, SubType

    out = {"iv_annual_pct": None, "spot": None, "iv_by_dte": [], "sigma2_by_dte": [], "interp_dtes": [], "interp_weights": [], "method": method, "error": None}
    cache_key = underlying.replace(".", "_") + f"_h{horizon}_" + date.today().isoformat() + f"_{method}"
    cache_path = os.path.join(data_dir, f"iv_{cache_key}.json")
    if os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("method") == method:
            out["iv_annual_pct"] = cached["iv_annual_pct"]
            out["spot"] = cached["spot"]
            out["iv_by_dte"] = [tuple(x) for x in cached["iv_by_dte"]]
            out["sigma2_by_dte"] = [tuple(x) for x in cached.get("sigma2_by_dte", [])]
            out["interp_dtes"] = cached.get("interp_dtes", [])
            out["interp_weights"] = cached.get("interp_weights", [])
            out["method"] = cached.get("method", method)
            return out

    own_ctx = quote_ctx is None
    if own_ctx:
        quote_ctx = OpenQuoteContext(host=host, port=port)

    try:
        ret_exp, df_exp = quote_ctx.get_option_expiration_date(code=underlying)
        if ret_exp != RET_OK:
            out["error"] = f"获取到期日失败: {df_exp}"
            if verbose:
                print(out["error"])
            return out

        df_exp = df_exp[
            (df_exp["option_expiry_date_distance"] > 0)
            & (df_exp["option_expiry_date_distance"] >= dte_range[0])
            & (df_exp["option_expiry_date_distance"] <= dte_range[1])
        ].head(6)
        if df_exp.empty:
            out["error"] = "无符合 DTE 范围的到期日"
            if verbose:
                print(out["error"])
            return out

        quote_ctx.subscribe([underlying], [SubType.QUOTE])
        ret_q, df_q = quote_ctx.get_stock_quote([underlying])
        spot = float(df_q["last_price"].iloc[0]) if ret_q == RET_OK else None

        dtes_all = sorted([int(r["option_expiry_date_distance"]) for _, r in df_exp.iterrows()])
        near = [t for t in dtes_all if t <= horizon]
        next_ = [t for t in dtes_all if t >= horizon]
        if near and next_:
            t1, t2 = max(near), min(next_)
        elif next_:
            t1, t2 = (next_[0], next_[1]) if len(next_) > 1 else (next_[0], next_[0])
        else:
            t1, t2 = (near[-2], near[-1]) if len(near) > 1 else (near[0], near[0])

        exp_dates = {int(r["option_expiry_date_distance"]): r["strike_time"] for _, r in df_exp.iterrows()}
        date1 = exp_dates.get(t1)
        date2 = exp_dates.get(t2)
        if not date1:
            date1 = df_exp.iloc[0]["strike_time"]
        if not date2 or t2 == t1:
            date2 = date1

        sigma2_list = []
        iv_list = []
        dte_chains = {}
        all_codes = []

        for dte, exp_date in [(t1, date1), (t2, date2)]:
            if dte in dte_chains:
                continue
            ret_chain, df_chain = quote_ctx.get_option_chain(code=underlying, start=exp_date, end=exp_date)
            if ret_chain != RET_OK or df_chain.empty:
                continue
            codes = df_chain["code"].tolist()
            dte_chains[dte] = (df_chain, codes)
            all_codes.extend(codes)

        if not all_codes:
            out["error"] = "无可用期权"
            if verbose:
                print(out["error"])
            return out

        quote_ctx.subscribe(all_codes, [SubType.QUOTE])
        ret_snap, df_snap = quote_ctx.get_market_snapshot(all_codes)
        if ret_snap != RET_OK or not hasattr(df_snap, "columns"):
            out["error"] = f"快照获取失败: {df_snap}" if ret_snap != RET_OK else "快照返回非 DataFrame"
            if verbose:
                print(out["error"])
            return out
        iv_col = "option_implied_volatility" if "option_implied_volatility" in df_snap.columns else "wrt_implied_volatility"

        use_vix = method == "vix_style"
        dtes_to_process = list(dict.fromkeys([t1, t2]))
        for dte in dtes_to_process:
            if dte not in dte_chains:
                continue
            df_chain, codes = dte_chains[dte]
            sub_snap = df_snap[df_snap["code"].isin(codes)]
            if use_vix and "bid_price" in sub_snap.columns and "ask_price" in sub_snap.columns:
                sigma2 = _compute_variance_vix_style(df_chain, sub_snap, spot, dte, risk_free_rate)
                if sigma2 is not None:
                    sigma2_list.append((dte, sigma2))
                    iv_list.append((dte, float(np.sqrt(sigma2) * 100)))
                    continue
            use_vix = False
            if iv_col in sub_snap.columns:
                strikes = df_chain["strike_price"].unique()
                atm = min(strikes, key=lambda x: abs(x - spot)) if spot else strikes[len(strikes) // 2]
                df_atm = df_chain[df_chain["strike_price"] == atm]
                atm_codes = df_atm["code"].tolist()
                atm_ivs = sub_snap[sub_snap["code"].isin(atm_codes)][iv_col].dropna()
                if len(atm_ivs) > 0:
                    iv_list.append((dte, float(atm_ivs.mean())))

        if t1 == t2:
            iv_30d = iv_list[0][1] if iv_list else None
            w1, w2 = 1.0, 0.0
            if iv_30d is None:
                out["error"] = "无有效 IV 数据"
                if verbose:
                    print(out["error"])
                return out
        elif use_vix and len(sigma2_list) >= 2:
            sigma2_by_dte = dict(sigma2_list)
            v1 = sigma2_by_dte[t1] * (t1 / 365)
            v2 = sigma2_by_dte[t2] * (t2 / 365)
            w1 = (t2 - horizon) / (t2 - t1)
            w2 = (horizon - t1) / (t2 - t1)
            v_horizon = w1 * v1 + w2 * v2
            iv_30d = float(np.sqrt(v_horizon * 365 / horizon) * 100)
            out["sigma2_by_dte"] = sigma2_list
        else:
            use_vix = False
            iv_by_dte = dict(iv_list)
            if len(iv_list) < 2:
                out["error"] = "至少需要两个到期日用于插值"
                if verbose:
                    print(out["error"])
                return out
            v1 = (iv_by_dte[t1] / 100) ** 2 * (t1 / 365)
            v2 = (iv_by_dte[t2] / 100) ** 2 * (t2 / 365)
            w1 = (t2 - horizon) / (t2 - t1)
            w2 = (horizon - t1) / (t2 - t1)
            v_horizon = w1 * v1 + w2 * v2
            iv_30d = float(np.sqrt(v_horizon * 365 / horizon) * 100)

        if not own_ctx:
            time.sleep(5)
            to_unsub = [underlying] + all_codes
            quote_ctx.unsubscribe(to_unsub, [SubType.QUOTE])

        out["iv_annual_pct"] = iv_30d
        out["spot"] = spot
        out["iv_by_dte"] = iv_list
        out["interp_dtes"] = [t1, t2]
        out["interp_weights"] = [w1, w2]
        out["method"] = "vix_style" if use_vix else "variance_interp"

        os.makedirs(data_dir, exist_ok=True)
        cache_data = {"iv_annual_pct": iv_30d, "spot": spot, "iv_by_dte": iv_list, "interp_dtes": out["interp_dtes"], "interp_weights": out["interp_weights"], "method": out["method"]}
        if out.get("sigma2_by_dte"):
            cache_data["sigma2_by_dte"] = out["sigma2_by_dte"]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

        if verbose:
            print(f"标的现价: {spot:.2f}")
            print(f"方法: {out['method']}")
            print(f"各到期日 IV (年化):")
            for d, iv in iv_list:
                print(f"  DTE={d:3d}: IV={iv:.2f}%")
            if out.get("sigma2_by_dte"):
                print(f"各到期日 sigma²: {dict(out['sigma2_by_dte'])}")
            print(f"插值到期: T1={t1}, T2={t2}, 权重=[{w1:.3f}, {w2:.3f}]")
            print(f"\n{horizon} 日整体 IV (年化): {iv_30d:.2f}%")

    finally:
        if own_ctx:
            quote_ctx.close()

    return out
