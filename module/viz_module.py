"""可视化模块。"""
import pandas as pd
import matplotlib.pyplot as plt


def plot_hv_garch(result: dict, symbol: str, figsize=(10, 5)):
    """绘制 HV 历史序列、GARCH 条件波动率历史及未来预测。"""
    last_dt = result["returns"].index[-1]
    h = result["horizon_days"]
    future_dates = [last_dt + pd.Timedelta(days=i) for i in range(1, h + 1)]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(result["hv_series"].index, result["hv_series"].values, label="HV, 30 rolling days", color="C0", lw=1.2)
    ax.plot(result["garch_vol_series"].index, result["garch_vol_series"].values, label="GARCH cond_vol_hist", color="C1", linestyle="--", lw=1.2)
    ax.plot(future_dates, result["annual_vol_pct"], label="GARCH pred_vol", color="C2", marker="o", ms=3, lw=1.2)
    ax.set_ylabel("annual_vol_pct")
    ax.set_xlabel("date")
    ax.set_title(f"{symbol}: HV and GARCH hist & pred")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
