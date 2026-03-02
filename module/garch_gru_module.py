"""
GARCH-GRU 混合波动率预测模块（arxiv 2504.09380）。

将 GARCH(1,1) 嵌入 GRU 单元，通过加法融合实现波动率聚类与非线性时序建模。

重要：输入必须为原始收益率（小数），不可标准化。GARCH 分量中 ε² 需与收益率
同单位，标准化会破坏尺度导致预测值异常放大（如 100%+ 年化）。
"""

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import (
    GARCH_BLEND_WEIGHT,
    GARCH_FORECAST_STYLE,
    GARCH_GRU_HIDDEN,
    GARCH_GRU_HORIZON_WEIGHTS,
    GARCH_GRU_LOSS,
    GARCH_GRU_RV_WINDOW,
    GARCH_GRU_TARGET,
    GARCH_GRU_VOL_CLAMP,
    GARCH_GRU_WINDOW,
    GARCH_HORIZON_DAYS,
    HOLD_PERIOD,
)


def _horizon_weights(horizon: int, style: str = "short") -> np.ndarray:
    """
    返回 horizon 维权重，用于加权 MSE。
    style="short": 1 日权重 2，3 日内 1.5，7 日内 1，其余 0.5
    style="equal": 全 1
    """
    if style == "equal":
        return np.ones(horizon, dtype=np.float32)
    if style == "short":
        w = np.ones(horizon, dtype=np.float32) * 0.5
        w[:7] = 1.0
        w[:3] = 1.5
        w[:1] = 2.0
        return w
    if isinstance(style, (list, tuple)) and len(style) == horizon:
        return np.array(style, dtype=np.float32)
    return np.ones(horizon, dtype=np.float32)


def _realized_volatility(returns: np.ndarray, window: int = 5) -> np.ndarray:
    """
    计算实现波动率：σt = sqrt(mean((r - μ)²))，k 日滚动。
    returns: 收益率序列（小数形式）
    """
    r = np.asarray(returns, dtype=float).flatten()
    n = len(r)
    rv = np.full(n, np.nan)
    for i in range(window - 1, n):
        block = r[i - window + 1 : i + 1]
        mu = np.mean(block)
        var = np.mean((block - mu) ** 2)
        rv[i] = np.sqrt(max(var, 1e-12))
    return rv


def _historical_volatility(returns: np.ndarray, window: int) -> np.ndarray:
    """
    计算历史波动率：rolling std，与图表 HV 一致。
    returns: 收益率序列（小数形式）
    """
    r = np.asarray(returns, dtype=float).flatten()
    hv = np.full(len(r), np.nan, dtype=float)
    for i in range(window - 1, len(r)):
        block = r[i - window + 1 : i + 1]
        hv[i] = np.std(block)
    return hv


def _build_sequences(
    returns: np.ndarray,
    rv: np.ndarray,
    window: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构造训练序列：(X, y, init_var)
    X: (N, window) 输入 returns（均值调整后）
    y: (N, horizon) 目标 realized vol，未来 horizon 日
    init_var: (N,) 每样本窗口样本方差，用于 GARCH 热启动
    """
    r = np.asarray(returns, dtype=float).flatten()
    rv = np.asarray(rv, dtype=float).flatten()
    n = len(r)
    # 有效起点：需要 window 个历史 + horizon 个未来
    start = window
    end = n - horizon
    if end <= start:
        return np.empty((0, window)), np.empty((0, horizon)), np.empty(0)

    X_list = []
    y_list = []
    init_var_list = []
    for i in range(start, end):
        block = r[i - window : i]
        mu = np.mean(block)
        # ε = r - μ（均值调整残差），与 GARCH(1,1) 及 RV 目标一致
        adj = block - mu
        X_list.append(adj)
        y_list.append(rv[i : i + horizon])
        # 窗口样本方差，用于 GARCH 热启动（无条件方差估计）
        init_var_list.append(max(np.mean(adj**2), 1e-12))
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(init_var_list, dtype=np.float32),
    )


class GARCHGRUCell(nn.Module):
    """
    GARCH-GRU 单元：将 GARCH(1,1) 嵌入 GRU，加法融合。
    公式：gt = φ(ω0 + α·ε² + β·σ²), ht = tanh(ĥt + γ·gt)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # GRU 门控参数
        self.Wz = nn.Linear(input_size, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size)
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.Wh = nn.Linear(input_size, hidden_size)
        self.Uh = nn.Linear(hidden_size, hidden_size)

        # GARCH 分量：φ(u) = Wg·u + bg，标量 -> hidden
        self.Wg = nn.Linear(1, hidden_size)

        # GARCH 参数（无约束，前向中 reparameterize）
        # omega 初始 ~1e-4，与典型日方差小数²（1e-4~1e-2）匹配
        self._omega_raw = nn.Parameter(torch.tensor(-8.0))
        # α、β 初值使 GARCH 从一开始就有明显作用，避免退化为纯 GRU
        self._alpha_raw = nn.Parameter(torch.tensor(1.0))
        self._beta_raw = nn.Parameter(torch.tensor(1.0))

        # γ：GARCH 贡献强度，提高初值强化波动率信号
        self.gamma = nn.Parameter(torch.tensor(0.4))

    def _garch_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reparameterize 保证 ω>0, α≥0, β≥0, α+β<1"""
        omega = torch.nn.functional.softplus(self._omega_raw) + 1e-6
        a = torch.sigmoid(self._alpha_raw)
        b = torch.sigmoid(self._beta_raw) * (1 - a * 0.99)  # α+β < 1
        return omega, a, b

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        eps_sq_prev: torch.Tensor,
        sigma_sq_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (batch, input_size) 当前收益
        h_prev: (batch, hidden_size)
        eps_sq_prev: (batch,) 上一时刻 ε²
        sigma_sq_prev: (batch,) 上一时刻 σ²
        返回 (h, eps_sq, sigma_sq)
        """
        omega, alpha, beta = self._garch_params()
        # gt = φ(ω + α·ε² + β·σ²)
        garch_in = omega + alpha * eps_sq_prev + beta * sigma_sq_prev
        garch_in = garch_in.unsqueeze(-1)  # (batch, 1)
        gt = self.Wg(garch_in)  # (batch, hidden)

        z = torch.sigmoid(self.Wz(x) + self.Uz(h_prev))
        r = torch.sigmoid(self.Wr(x) + self.Ur(h_prev))
        h_tilde = torch.tanh(self.Wh(x) + self.Uh(r * h_prev))
        h_hat = (1 - z) * h_tilde + z * h_prev
        h = torch.tanh(h_hat + self.gamma * gt)

        # 更新 ε², σ² 用于下一步
        eps = x[:, 0] if x.dim() > 1 and x.size(1) >= 1 else x.squeeze(-1)
        eps_sq = eps ** 2
        sigma_sq = garch_in.squeeze(-1)  # 标量 GARCH 输出即 σ²

        return h, eps_sq, sigma_sq


class GARCHGRUModel(nn.Module):
    """
    GARCH-GRU 波动率预测模型。
    输入：returns 序列 -> 输出：未来 horizon 日波动率（年化 %）
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        horizon: int = 22,
        dropout: float = 0.1,
        vol_clamp: tuple[float, float] | None = (0.01, 10.0),
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.vol_clamp = vol_clamp

        self.cell = GARCHGRUCell(input_size, hidden_size)
        # fc_out 输出 NN 缩放因子，与 GARCH vol_base 相乘得最终波动率
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
            nn.Softplus(),  # 缩放因子 ≥0，初始接近 0 使 vol≈vol_base
        )
        with torch.no_grad():
            self.fc_out[-2].bias.fill_(-5.0)  # Softplus(-5)≈0.007，1+scale≈1
            self.fc_out[-2].weight.data *= 0.1

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        eps_sq_0: torch.Tensor | None = None,
        sigma_sq_0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, input_size)
        返回 (vol, sigma_sq_final)
        vol: (batch, horizon) 日波动率（小数）
        sigma_sq_final: (batch,) 序列末 GARCH 条件方差
        """
        batch, seq_len, _ = x.shape
        device = x.device

        if h0 is None:
            h0 = torch.zeros(batch, self.hidden_size, device=device)
        if eps_sq_0 is None:
            eps_sq_0 = torch.full((batch,), 1e-6, device=device)
        if sigma_sq_0 is None:
            sigma_sq_0 = torch.full((batch,), 1e-6, device=device)

        h = h0
        eps_sq = eps_sq_0
        sigma_sq = sigma_sq_0

        for t in range(seq_len):
            xt = x[:, t, :]
            h, eps_sq, sigma_sq = self.cell(xt, h, eps_sq, sigma_sq)

        # 方案 A：vol = vol_base * (1 + nn_scale)，vol_base 随序列变化
        vol_base = torch.sqrt(sigma_sq + 1e-8)  # (batch,)
        nn_scale = self.fc_out(h)  # (batch, horizon)
        vol = vol_base.unsqueeze(-1) * (1.0 + nn_scale)
        if self.vol_clamp is not None:
            vol = torch.clamp(vol, self.vol_clamp[0], self.vol_clamp[1])
        return vol, sigma_sq


def _fit_garch_gru(
    returns: pd.Series,
    horizon_days: int = GARCH_HORIZON_DAYS,
    window: int = GARCH_GRU_WINDOW,
    rv_window: int = GARCH_GRU_RV_WINDOW,
    hidden_size: int = GARCH_GRU_HIDDEN,
    epochs: int = 200,
    patience: int = 20,
    lr: float = 1e-3,
    device: str | None = None,
    verbose: bool = False,
) -> tuple[GARCHGRUModel, np.ndarray, np.ndarray]:
    """
    训练 GARCH-GRU 并返回模型、条件波动率历史、预测序列。
    returns: 收益率（小数形式）
    返回 (model, cond_vol_hist, forecast_vol)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    r = returns.dropna().values.astype(np.float32)
    if GARCH_GRU_TARGET == "hv":
        vol_series = _historical_volatility(r, window=rv_window)
    else:
        vol_series = _realized_volatility(r, window=rv_window)

    X, y, init_var = _build_sequences(r, vol_series, window, horizon_days)
    if len(X) < 50:
        raise ValueError(
            f"GARCH-GRU 需要足够序列用于训练，当前仅 {len(X)} 条。请扩大样本区间。"
        )
    if len(X) < 400:
        warnings.warn(
            f"GARCH-GRU 样本量偏少（{len(X)} 条），建议至少 400+ 以提升稳定性。"
            "可考虑扩大 start_date 拉长历史（如 2018 年起）。",
            UserWarning,
            stacklevel=2,
        )

    # 全程小数：X、y、init_var 均为小数，不 scale
    X_input = X
    y_input = y
    init_var_input = init_var

    # 划分 train/val（最后 20% 作验证）
    n = len(X_input)
    split = int(n * 0.8)
    X_train, y_train = X_input[:split], y_input[:split]
    X_val, y_val = X_input[split:], y_input[split:]

    X_train_t = torch.from_numpy(X_train).unsqueeze(-1).to(device)  # (N, window, 1)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).unsqueeze(-1).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    init_var_train = torch.from_numpy(init_var_input[:split]).to(device)
    init_var_val = torch.from_numpy(init_var_input[split:]).to(device)

    model = GARCHGRUModel(
        input_size=1,
        hidden_size=hidden_size,
        horizon=horizon_days,
        dropout=0.1,
        vol_clamp=GARCH_GRU_VOL_CLAMP,
    ).to(device)
    horizon_w = _horizon_weights(horizon_days, GARCH_GRU_HORIZON_WEIGHTS)
    horizon_w_t = torch.from_numpy(horizon_w).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )

    if verbose:
        print(f"GARCH-GRU: n_train={len(X_train)}, n_val={len(X_val)}, horizon={horizon_days}")

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for ep in range(epochs):
        model.train()
        pred, _ = model(
            X_train_t,
            eps_sq_0=init_var_train,
            sigma_sq_0=init_var_train,
        )
        # 加权 loss：log_mse 抑制塌缩，mse 标准
        eps = 1e-6
        if GARCH_GRU_LOSS == "log_mse":
            log_err = (torch.log(pred + eps) - torch.log(y_train_t + eps)) ** 2
            loss = (log_err * horizon_w_t).mean()
        else:
            sq_err = (pred - y_train_t) ** 2
            loss = (sq_err * horizon_w_t).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred, _ = model(
                X_val_t,
                eps_sq_0=init_var_val,
                sigma_sq_0=init_var_val,
            )
            _eps = 1e-6
            if GARCH_GRU_LOSS == "log_mse":
                val_log_err = (torch.log(val_pred + _eps) - torch.log(y_val_t + _eps)) ** 2
                val_loss = (val_log_err * horizon_w_t).mean().item()
            else:
                val_sq_err = (val_pred - y_val_t) ** 2
                val_loss = (val_sq_err * horizon_w_t).mean().item()
        scheduler.step(val_loss)

        if verbose and (ep == 0 or (ep + 1) % 20 == 0 or val_loss < best_val_loss):
            if GARCH_GRU_LOSS == "log_mse":
                train_loss = (log_err * horizon_w_t).mean().item()
            else:
                train_loss = (sq_err * horizon_w_t).mean().item()
            sample_pred = pred[0].detach().cpu().numpy()[:5]
            sample_y = y_train_t[0].cpu().numpy()[:5]
            print(
                f"  ep {ep+1}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"| pred[:5]={sample_pred.round(4).tolist()} y[:5]={sample_y.round(4).tolist()}"
            )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  early stop at ep {ep+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # 计算 in-sample 条件波动率（用于 garch_vol_series）
    with torch.no_grad():
        full_X = torch.from_numpy(X_input).unsqueeze(-1).to(device)
        full_init_var = torch.from_numpy(init_var_input).to(device)
        full_pred, _ = model(
            full_X,
            eps_sq_0=full_init_var,
            sigma_sq_0=full_init_var,
        )
        full_pred = full_pred.cpu().numpy()

    # cond_vol_hist：vol = vol_base * (1+nn_scale)，已含 GARCH 时变
    cond_vol_hist = np.full(len(r), np.nan, dtype=float)
    valid_start = window
    valid_end = valid_start + len(full_pred)
    cond_vol_hist[valid_start:valid_end] = full_pred[:, 0]

    # 最后一段用最近窗口预测
    last_X = X_input[-1:].copy()
    last_t = torch.from_numpy(last_X).unsqueeze(-1).to(device)
    last_init_var = torch.from_numpy(init_var_input[-1:]).to(device)
    with torch.no_grad():
        forecast, _ = model(
            last_t,
            eps_sq_0=last_init_var,
            sigma_sq_0=last_init_var,
        )
        forecast = forecast.cpu().numpy().squeeze(0)

    # 模型输出已是小数
    return model, cond_vol_hist, forecast


def get_garch_gru_volatility(
    returns: pd.Series,
    garch_horizon_days: int = GARCH_HORIZON_DAYS,
    hold_period: float = HOLD_PERIOD,
    hv_annual_pct: float = 0.0,
    verbose: bool = False,
) -> dict:
    """
    使用 GARCH-GRU 计算波动率指标，返回与 get_volatility_metrics 兼容的 dict 子集。
    包含：cond_vol_pct, annual_vol_pct, garch_1d_pct, garch_nd_pct, garch_vol_series,
    cond_vol_hist, model_spec, forecast_style
    """
    _, cond_vol_hist, forecast_vol = _fit_garch_gru(
        returns,
        horizon_days=garch_horizon_days,
        window=GARCH_GRU_WINDOW,
        rv_window=GARCH_GRU_RV_WINDOW,
        hidden_size=GARCH_GRU_HIDDEN,
        verbose=verbose,
    )

    # forecast_vol: (horizon,) 日波动率（小数）
    # cond_var 为小数²，与 arch 转换后一致，供 garch_module 计算 expected_move
    cond_var = forecast_vol ** 2
    cond_vol_pct = forecast_vol * 100  # 日波动率 %（仅展示）
    annual_vol_pct = cond_vol_pct * np.sqrt(252)  # 年化 %

    garch_1d_pct = float(annual_vol_pct[0])

    style = GARCH_FORECAST_STYLE
    if style == "terminal":
        garch_nd_pct = float(annual_vol_pct[-1])
    elif style == "average":
        avg_var = np.mean(cond_var)  # 小数²
        garch_nd_pct = float(np.sqrt(avg_var) * np.sqrt(252) * 100)
    elif style == "blend":
        avg_var = np.mean(cond_var)  # 小数²
        garch_avg = np.sqrt(avg_var) * np.sqrt(252) * 100
        garch_nd_pct = float(GARCH_BLEND_WEIGHT * garch_avg + (1 - GARCH_BLEND_WEIGHT) * hv_annual_pct)
    else:
        garch_nd_pct = float(annual_vol_pct[-1])

    # garch_vol_series：与 arch 的 conditional_volatility * sqrt(252) 对齐
    returns_clean = returns.dropna()
    idx = returns_clean.index
    garch_vol_series = pd.Series(cond_vol_hist * np.sqrt(252) * 100, index=idx)

    return {
        "cond_vol_pct": cond_vol_pct,
        "annual_vol_pct": annual_vol_pct,
        "garch_1d_pct": garch_1d_pct,
        "garch_nd_pct": garch_nd_pct,
        "garch_vol_series": garch_vol_series,
        "cond_vol_hist": pd.Series(cond_vol_hist, index=returns_clean.index),
        "model_spec": ("GARCH-GRU", "PyTorch"),
        "forecast_style": style,
        "cond_var": cond_var,
    }
