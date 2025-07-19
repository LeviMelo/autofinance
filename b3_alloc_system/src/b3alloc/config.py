from pathlib import Path
from typing import Literal, Optional, Dict

import yaml
from pydantic import BaseModel, Field, field_validator

# Pydantic models provide data validation, type hints, and a clear structure
# that mirrors the YAML configuration files specified in the project blueprint.

# --- Nested Models for Configuration Sections ---

class DataConfig(BaseModel):
    """Configuration for data sources and parameters."""
    start: str
    end: str
    tickers_file: str
    selic_series: int
    publish_lag_days: int = Field(..., gt=0) # Must be a positive integer

class GarchConfig(BaseModel):
    """Configuration for GARCH model in the Risk Engine."""
    dist: Literal["gaussian", "student-t"] = "gaussian"
    min_obs: int = Field(500, gt=252)
    refit_freq_days: int = Field(21, gt=0)

class DccConfig(BaseModel):
    """Configuration for DCC model in the Risk Engine."""
    a_init: float = Field(..., gt=0, lt=1)
    b_init: float = Field(..., gt=0, lt=1)

class ShrinkageConfig(BaseModel):
    """Configuration for covariance shrinkage."""
    method: Literal["ledoit_wolf", "ledoit_wolf_constant_corr", "manual"]
    floor: float = Field(0.0, ge=0, lt=1)

class RiskEngineConfig(BaseModel):
    """Configuration group for the entire Risk Engine."""
    garch: GarchConfig
    dcc: DccConfig
    shrinkage: ShrinkageConfig

class FactorViewConfig(BaseModel):
    """Configuration for the Fama-French return view."""
    lookback_days: int = Field(..., gt=252)
    include_alpha: bool
    premium_estimator: Literal["long_term_mean", "expanding_mean", "ewm"]

class VarViewConfig(BaseModel):
    """Configuration for the VAR return view."""
    max_lag: int = Field(..., gt=0, le=20)
    criterion: Literal["aic", "bic"]
    log_returns: bool

class ReturnEngineConfig(BaseModel):
    """Configuration group for the Return Engine."""
    factor: FactorViewConfig
    var: VarViewConfig

class BLConfidenceConfig(BaseModel):
    """Configuration for Black-Litterman view confidences (Omega matrix)."""
    method: Literal["rmse_based", "user_scaled"]
    factor_scaler: float = Field(1.0, gt=0)
    var_scaler: float = Field(1.0, gt=0)

class BlackLittermanConfig(BaseModel):
    """Configuration for the Black-Litterman synthesizer."""
    tau: float = Field(..., gt=0, lt=1)
    confidence: BLConfidenceConfig

class OptimizerConfig(BaseModel):
    """Configuration for the portfolio optimizer."""
    objective: Literal["max_sharpe", "min_variance", "target_return", "target_vol"]
    long_only: bool
    name_cap: float = Field(..., ge=0.01, le=1.0)
    sector_cap: float = Field(..., ge=0.01, le=1.0)
    turnover_penalty_bps: int = Field(..., ge=0)

class BacktestConfig(BaseModel):
    """Configuration for the backtesting engine."""
    lookback_years: int = Field(..., gt=1)
    rebalance: Literal["monthly", "quarterly"]
    start: str
    end: str
    costs_bps: int = Field(..., ge=0)

# --- Top-Level Configuration Model ---

class Config(BaseModel):
    """The main configuration object, loading all sub-sections."""
    data: DataConfig
    risk_engine: RiskEngineConfig
    return_engine: ReturnEngineConfig
    black_litterman: BlackLittermanConfig
    optimizer: OptimizerConfig
    backtest: BacktestConfig
    
    # Per Amendment 1
    # These fields are optional to maintain backwards compatibility.
    universe: Optional[Dict] = None
    tax: Optional[Dict] = None
    meta: Optional[Dict] = None


def load_config(config_path: str | Path) -> Config:
    """
    Loads and validates the master YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        A validated Config object.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)


if __name__ == '__main__':
    # This block is for demonstrating and testing the config loading.
    # It requires creating a sample YAML file.

    # 1. Create a dummy YAML for testing
    dummy_yaml_content = """
data:
  start: 2010-01-01
  end: 2025-01-01
  tickers_file: config/universe_small.csv
  selic_series: 11
  publish_lag_days: 3

risk_engine:
  garch:
    dist: gaussian
    min_obs: 500
    refit_freq_days: 21
  dcc:
    a_init: 0.02
    b_init: 0.97
  shrinkage:
    method: ledoit_wolf_constant_corr
    floor: 0.05

return_engine:
  factor:
    lookback_days: 756
    include_alpha: false
    premium_estimator: long_term_mean
  var:
    max_lag: 5
    criterion: bic
    log_returns: true

black_litterman:
  tau: 0.05
  confidence:
    method: rmse_based
    factor_scaler: 1.0
    var_scaler: 0.5

optimizer:
  objective: max_sharpe
  long_only: true
  name_cap: 0.10
  sector_cap: 0.25
  turnover_penalty_bps: 5

backtest:
  lookback_years: 5
  rebalance: monthly
  start: 2012-01-01
  end: 2025-01-01
  costs_bps: 10
"""
    # Create a temporary directory and file
    temp_dir = Path("./temp_config")
    temp_dir.mkdir(exist_ok=True)
    dummy_config_path = temp_dir / "test_config.yaml"
    with open(dummy_config_path, "w") as f:
        f.write(dummy_yaml_content)

    # 2. Test loading the config
    print("--- Loading Test Config ---")
    try:
        config = load_config(dummy_config_path)
        print("Configuration loaded and validated successfully!")
        
        # 3. Access nested parameters
        print("\n--- Accessing Parameters ---")
        print(f"Project Data Start Date: {config.data.start}")
        print(f"Risk Engine Shrinkage Method: {config.risk_engine.shrinkage.method}")
        print(f"Black-Litterman Tau: {config.black_litterman.tau}")
        print(f"Optimizer Name Cap: {config.optimizer.name_cap}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Clean up the dummy file and directory
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        print("\nCleaned up temporary config files.")