"""
集中配置常量，便于维护与修改。
"""

# 数据与缓存
DATA_DIR = "./data"

# 富途 OpenD 连接
FUTU_HOST = "127.0.0.1"
FUTU_PORT = 11111

# IV 计算
IV_HORIZON_DAYS = 30
IV_DTE_RANGE = (7, 60)
IV_METHOD = "vix_style"  # "vix_style" | "variance_interp"
RISK_FREE_RATE = 0.04

# GARCH 与波动率
GARCH_HORIZON_DAYS = 22
HOLD_PERIOD = 1

# 数据获取
DEFAULT_START_DATE = "2010-01-01"
MAX_RETRIES = 3
RETRY_DELAY = 5
DEFAULT_PROVIDER = "yfinance"

# 示例运行（可覆盖）
# GARCH 需至少 252 个交易日（约 1 年），建议 START 不晚于 2024-01-01
SYMBOL = "TSM"
START = "2020-01-01"
END = None
