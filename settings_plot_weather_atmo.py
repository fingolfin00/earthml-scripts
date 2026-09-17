from earthml import (
    SeqWRdY,
    PiBRdY,
)


# =============================================================================
# Absolute metric plotting overrides
#
# Anything not specified here falls back to DEFAULT_PLOT_CONFIG.
# =============================================================================

VARIABLE_PLOT_CONFIG = {
    # =========================================================================
    # Mean Sea Level Pressure [hPa after plotting conversion]
    # =========================================================================
    "mslp": {
        "bias": {"vmin": -1, "vmax": 1, "ticks": [-1, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 1]},
        "mae": {"vmin": 0, "vmax": 3, "ticks": 11},
        "mse": {"vmin": 0, "vmax": 9, "ticks": [0, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
        "rmse": {"vmin": 0, "vmax": 3, "ticks": 11},
        "nrmse": {"vmin": 0, "vmax": 0.3, "ticks": 11},

        "bias_anom": {"vmin": -1, "vmax": 1, "ticks": [-1, -0.5, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.5, 1]},
        "mae_anom": {"vmin": 0, "vmax": 3, "ticks": 11},
        "mse_anom": {"vmin": 0, "vmax": 9, "ticks": [0, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
        "rmse_anom": {"vmin": 0, "vmax": 3, "ticks": 11},
        "nrmse_anom": {"vmin": 0, "vmax": 0.5, "ticks": 11},

        "corr": {"vmin": 0.94, "vmax": 1, "ticks": 11, "cmap": SeqWRdY},
        "acc": {"vmin": 0.95, "vmax": 1, "ticks": 11, "cmap": SeqWRdY},
        "r2": {"vmin": 0.9, "vmax": 1, "ticks": 11, "cmap": SeqWRdY},

        "fc_std": {"vmin": 0, "vmax": 14, "ticks": 11},
        "an_std": {"vmin": 0, "vmax": 14, "ticks": 11},
        "fc_anom_std": {"vmin": 0, "vmax": 14, "ticks": 11},
        "an_anom_std": {"vmin": 0, "vmax": 14, "ticks": 11},

        # Target = 1: use default PiBRdY / TwoSlopeNorm(center=1).
        "std_ratio": {"vmin": 0.9, "vmax": 1.1, "ticks": 11},
        "std_ratio_anom": {"vmin": 0.9, "vmax": 1.1, "ticks": 11},
        "regression_slope": {"vmin": 0.9, "vmax": 1.1, "ticks": 11},
        "regression_slope_anom": {"vmin": 0.9, "vmax": 1.1, "ticks": 11},

        "mse_bias_component": {"vmin": 0, "vmax": 0.5, "ticks": 11},
        "mse_std_component": {"vmin": 0, "vmax": 0.5, "ticks": 11},
        "mse_corr_component": {"vmin": 0, "vmax": 7, "ticks": 11},
        "crmse": {"vmin": 0, "vmax": 3, "ticks": 11},

        "mse_bias_component_anom": {"vmin": 0, "vmax": 0.5, "ticks": 11},
        "mse_std_component_anom": {"vmin": 0, "vmax": 0.5, "ticks": 11},
        "mse_corr_component_anom": {"vmin": 0, "vmax": 7, "ticks": 11},
        "crmse_anom": {"vmin": 0, "vmax": 3, "ticks": 11},

        "mse_skill_clim": {"vmin": 0.8, "vmax": 1, "ticks": 11, "cmap": SeqWRdY, "scale_units": False},

        "ens_member_rmse": {"vmin": 0, "vmax": 3, "ticks": 11},
        "mean_member_rmse": {"vmin": 0, "vmax": 3, "ticks": 11},
        "ens_member_rmse_anom": {"vmin": 0, "vmax": 3, "ticks": 11},
        "mean_member_rmse_anom": {"vmin": 0, "vmax": 3, "ticks": 11},
        "spread": {"vmin": 0, "vmax": 14, "ticks": 11},
        "spread_anom": {"vmin": 0, "vmax": 14, "ticks": 11},
        "spread_skill_ratio": {"vmin": 0.5, "vmax": 1.5, "ticks": [0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5]},
        "spread_anom_skill_ratio": {"vmin": 0.5, "vmax": 1.5, "ticks": [0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5]},
        "crps": {"vmin": 0, "vmax": 3, "ticks": 11},
        "crps_anom": {"vmin": 0, "vmax": 3, "ticks": 11},

        "fc_grad_mag": {"vmin": 0, "vmax": 0.006, "ticks": [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006]},
        "an_grad_mag": {"vmin": 0, "vmax": 0.006, "ticks": [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006]},
        "grad_rmse": {"vmin": 0, "vmax": 0.004, "ticks": [0, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004]},
        "fc_anom_grad_mag": {"vmin": 0, "vmax": 0.004, "ticks": [0, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004]},
        "an_anom_grad_mag": {"vmin": 0, "vmax": 0.004, "ticks": [0, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004]},
        "grad_rmse_anom": {"vmin": 0, "vmax": 0.003, "ticks": [0, 0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.002, 0.0025, 0.003]},
    },

    # =========================================================================
    # 2 m Temperature
    # =========================================================================
    "t2m": {
        "bias": {"vmin": -4, "vmax": 4, "ticks": [-4, -3, -2.5, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4]},
        "mae": {"vmin": 0, "vmax": 4, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4]},
        "mse": {"vmin": 0, "vmax": 16, "ticks": [0, 0.25, 1, 2.25, 4, 6.25, 9, 12.25, 16]},
        "rmse": {"vmin": 0, "vmax": 4, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4]},
        "bias_anom": {"vmin": -3, "vmax": 3, "ticks": [-3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3]},
        "mae_anom": {"vmin": 0, "vmax": 4, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4]},
        "mse_anom": {"vmin": 0, "vmax": 9, "ticks": [0, 0.25, 1, 2.25, 4, 6.25, 9]},
        "rmse_anom": {"vmin": 0, "vmax": 3, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]},
        "std_ratio": {"vmin": 0.75, "vmax": 1.25, "ticks": [0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25]},
        "std_ratio_anom": {"vmin": 0.75, "vmax": 1.25, "ticks": [0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25]},
        "regression_slope": {"vmin": 0.75, "vmax": 1.25, "ticks": [0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25]},
        "regression_slope_anom": {"vmin": 0.75, "vmax": 1.25, "ticks": [0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25]},
        "crmse": {"vmin": 0, "vmax": 4, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4]},
        "crmse_anom": {"vmin": 0, "vmax": 3, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]},
    },

    # =========================================================================
    # 2 m Dew Point
    # =========================================================================
    "d2m": {
        "bias": {"vmin": -6, "vmax": 6, "ticks": [-6, -4, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 4, 6]},
        "mae": {"vmin": 0, "vmax": 6, "ticks": [0, 0.5, 1, 1.5, 2, 4, 6]},
        "mse": {"vmin": 0, "vmax": 36, "ticks": [0, 1, 4, 9, 16, 25, 36]},
        "rmse": {"vmin": 0, "vmax": 6, "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6]},
        "mae_anom": {"vmin": 0, "vmax": 5, "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]},
        "mse_anom": {"vmin": 0, "vmax": 25, "ticks": [0, 1, 4, 9, 16, 25]},
        "rmse_anom": {"vmin": 0, "vmax": 5, "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]},
        "mse_bias_component": {"vmin": 0, "vmax": 16, "ticks": [0, 1, 2, 4, 6, 8, 10, 12, 14, 16]},
        "mse_std_component": {"vmin": 0, "vmax": 16, "ticks": [0, 1, 2, 4, 6, 8, 10, 12, 14, 16]},
        "mse_corr_component": {"vmin": 0, "vmax": 16, "ticks": [0, 1, 2, 4, 6, 8, 10, 12, 14, 16]},
        "crmse": {"vmin": 0, "vmax": 4, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4]},
        "mse_bias_component_anom": {"vmin": 0, "vmax": 9, "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
        "mse_std_component_anom": {"vmin": 0, "vmax": 9, "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
        "mse_corr_component_anom": {"vmin": 0, "vmax": 9, "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
        "crmse_anom": {"vmin": 0, "vmax": 3, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]},
    },

    # =========================================================================
    # 10 m Wind
    # =========================================================================
    "u10": {
        "bias": {"vmin": -3, "vmax": 3, "ticks": [-3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3]},
        "mae": {"vmin": 0, "vmax": 5, "ticks": [0, 0.5, 1, 2, 3, 4, 5]},
        "mse": {"vmin": 0, "vmax": 36, "ticks": [0, 1, 4, 9, 16, 25, 36]},
        "rmse": {"vmin": 0, "vmax": 6, "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6]},
        "mae_anom": {"vmin": 0, "vmax": 2, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2]},
        "mse_anom": {"vmin": 0, "vmax": 4, "ticks": [0, 0.25, 0.5, 1, 2, 3, 4]},
        "rmse_anom": {"vmin": 0, "vmax": 2, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2]},
        "std_ratio": {"vmin": 0.7, "vmax": 1.3, "ticks": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]},
        "std_ratio_anom": {"vmin": 0.7, "vmax": 1.3, "ticks": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]},
    },

    "v10": {
        "bias": {"vmin": -3, "vmax": 3, "ticks": [-3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3]},
        "mae": {"vmin": 0, "vmax": 5, "ticks": [0, 0.5, 1, 2, 3, 4, 5]},
        "mse": {"vmin": 0, "vmax": 36, "ticks": [0, 1, 4, 9, 16, 25, 36]},
        "rmse": {"vmin": 0, "vmax": 6, "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6]},
        "mae_anom": {"vmin": 0, "vmax": 2, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2]},
        "mse_anom": {"vmin": 0, "vmax": 4, "ticks": [0, 0.25, 0.5, 1, 2, 3, 4]},
        "rmse_anom": {"vmin": 0, "vmax": 2, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2]},
        "std_ratio": {"vmin": 0.7, "vmax": 1.3, "ticks": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]},
        "std_ratio_anom": {"vmin": 0.7, "vmax": 1.3, "ticks": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]},
    },

    # =========================================================================
    # Sea Surface Temperature
    # Values below are a reasonable first pass; tune from actual maps.
    # =========================================================================
    "sst": {
        "bias": {"vmin": -2, "vmax": 2, "ticks": [-2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2]},
        "mae": {"vmin": 0, "vmax": 3, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]},
        "mse": {"vmin": 0, "vmax": 9, "ticks": [0, 0.25, 1, 2.25, 4, 6.25, 9]},
        "rmse": {"vmin": 0, "vmax": 3, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]},
        "bias_anom": {"vmin": -1.5, "vmax": 1.5, "ticks": [-1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5]},
        "mae_anom": {"vmin": 0, "vmax": 2, "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1, 1.25, 1.5, 2]},
        "mse_anom": {"vmin": 0, "vmax": 4, "ticks": [0, 0.25, 0.5, 1, 2, 3, 4]},
        "rmse_anom": {"vmin": 0, "vmax": 2, "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1, 1.25, 1.5, 2]},
        "std_ratio": {"vmin": 0.75, "vmax": 1.25, "ticks": [0.75, 0.8, 0.9, 1, 1.1, 1.2, 1.25]},
        "std_ratio_anom": {"vmin": 0.75, "vmax": 1.25, "ticks": [0.75, 0.8, 0.9, 1, 1.1, 1.2, 1.25]},
    },

    # =========================================================================
    # Precipitation rate
    # Highly distribution-dependent; keep broad.
    # =========================================================================
    "tprate": {
        "bias": {"vmin": -4, "vmax": 4, "ticks": [-4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4]},
        "mae": {"vmin": 0, "vmax": 8, "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 8]},
        "rmse": {"vmin": 0, "vmax": 10, "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10]},
        "rmse_anom": {"vmin": 0, "vmax": 8, "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 8]},
    },

    # =========================================================================
    # Total Cloud Cover [fraction]
    # =========================================================================
    "tcc": {
        "bias": {"vmin": -0.1, "vmax": 0.1, "ticks": [-0.1, -0.07, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07, 0.1]},
        "mae": {"vmin": 0, "vmax": 0.6, "ticks": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]},
        "mse": {"vmin": 0, "vmax": 0.36, "ticks": [0, 0.0025, 0.01, 0.0225, 0.04, 0.09, 0.16, 0.25, 0.36]},
        "rmse": {"vmin": 0, "vmax": 0.6, "ticks": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]},
        "mae_anom": {"vmin": 0, "vmax": 0.6, "ticks": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]},
        "mse_anom": {"vmin": 0, "vmax": 0.36, "ticks": [0, 0.0025, 0.01, 0.0225, 0.04, 0.09, 0.16, 0.25, 0.36]},
        "rmse_anom": {"vmin": 0, "vmax": 0.6, "ticks": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]},
    },

    # =========================================================================
    # Ocean Mixed Layer Thickness
    # =========================================================================
    "mlotst": {
        "bias": {"vmin": -30, "vmax": 30, "ticks": [-30, -20, -10, -5, -2, 2, 5, 10, 20, 30]},
        "mae": {"vmin": 0, "vmax": 30, "ticks": [0, 2, 5, 10, 15, 20, 25, 30]},
        "rmse": {"vmin": 0, "vmax": 40, "ticks": [0, 2, 5, 10, 15, 20, 25, 30, 40]},
        "rmse_anom": {"vmin": 0, "vmax": 30, "ticks": [0, 2, 5, 10, 15, 20, 25, 30]},
    },

    # =========================================================================
    # Sea Surface Height
    # =========================================================================
    "ssh": {
        "bias": {"vmin": -0.5, "vmax": 0.5, "ticks": [-0.5, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.5]},
        "mae": {"vmin": 0, "vmax": 0.5, "ticks": [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]},
        "rmse": {"vmin": 0, "vmax": 0.5, "ticks": [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]},
        "rmse_anom": {"vmin": 0, "vmax": 0.4, "ticks": [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]},
    },

    # =========================================================================
    # Sea Surface Salinity
    # =========================================================================
    "sss": {
        "bias": {"vmin": -2, "vmax": 2, "ticks": [-2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2]},
        "mae": {"vmin": 0, "vmax": 2, "ticks": [0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]},
        "rmse": {"vmin": 0, "vmax": 2, "ticks": [0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]},
        "rmse_anom": {"vmin": 0, "vmax": 1.5, "ticks": [0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]},
    },

    # =========================================================================
    # Temperature at 20 m
    # =========================================================================
    "t20d": {
        "bias": {"vmin": -4, "vmax": 4, "ticks": [-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4]},
        "mae": {"vmin": 0, "vmax": 4, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4]},
        "mse": {"vmin": 0, "vmax": 16, "ticks": [0, 0.25, 1, 2.25, 4, 6.25, 9, 12.25, 16]},
        "rmse": {"vmin": 0, "vmax": 4, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4]},
        "rmse_anom": {"vmin": 0, "vmax": 3, "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]},
    },
}


# =============================================================================
# Improvement plotting overrides
#
# Anything not specified here falls back to DEFAULT_IMPROVEMENT_PLOT_CONFIG
# for the representation selected by METRIC_IMPROVEMENT_UNITS.
# Positive = ML improvement.
# =============================================================================

IMPROVEMENT_PLOT_CONFIG = {
    # =========================================================================
    # Mean Sea Level Pressure
    # =========================================================================
    "mslp": {
        "bias": {
            "%": {"vmin": -400, "vmax": 100, "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -1, "vmax": 1, "ticks": 21},
            "normalized": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },
        "mae": {
            "%": {"vmin": -30, "vmax": 30, "ticks": 11},
            "Δ": {"vmin": -1, "vmax": 1, "ticks": 21},
            "normalized": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },
        "mse": {
            "%": {"vmin": -50, "vmax": 50, "ticks": 11},
            "Δ": {"vmin": -2, "vmax": 2, "ticks": 11},
            "normalized": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
        },
        "rmse": {
            "%": {"vmin": -30, "vmax": 30, "ticks": 11},
            "Δ": {"vmin": -1, "vmax": 1, "ticks": 21},
            "normalized": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },
        "nrmse": {
            "%": {"vmin": -30, "vmax": 30, "ticks": 11},
            "Δ": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },

        "bias_anom": {
            "%": {"vmin": -400, "vmax": 100, "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -0.5, "vmax": 0.5, "ticks": 11},
            "normalized": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },
        "mae_anom": {
            "%": {"vmin": -30, "vmax": 30, "ticks": 11},
            "Δ": {"vmin": -0.5, "vmax": 0.5, "ticks": 11},
            "normalized": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },
        "mse_anom": {
            "%": {"vmin": -50, "vmax": 50, "ticks": 11},
            "Δ": {"vmin": -1, "vmax": 1, "ticks": 11},
            "normalized": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
        },
        "rmse_anom": {
            "%": {"vmin": -30, "vmax": 30, "ticks": 11},
            "Δ": {"vmin": -0.5, "vmax": 0.5, "ticks": 11},
            "normalized": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },
        "nrmse_anom": {
            "%": {"vmin": -30, "vmax": 30, "ticks": 11},
            "Δ": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
        },

        "corr": {"Δ": {"vmin": -0.01, "vmax": 0.01, "ticks": 11}},
        "acc": {"Δ": {"vmin": -0.02, "vmax": 0.02, "ticks": 11}},
        "r2": {"Δ": {"vmin": -0.1, "vmax": 0.1, "ticks": 11}},
        "r2_anom": {"Δ": {"vmin": -0.1, "vmax": 0.1, "ticks": 11}},

        "std_ratio": {
            "%": {"vmin": -200, "vmax": 100, "ticks": [-200, -150, -100, -50, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },
        "std_ratio_anom": {
            "%": {"vmin": -200, "vmax": 100, "ticks": [-200, -150, -100, -50, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },
        "regression_slope": {
            "%": {"vmin": -400, "vmax": 100, "ticks": [-400, -200, -150, -100, -50, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -0.05, "vmax": 0.05, "ticks": 11},
        },
        "regression_slope_anom": {
            "%": {"vmin": -400, "vmax": 100, "ticks": [-400, -200, -150, -100, -50, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -0.05, "vmax": 0.05, "ticks": 11},
        },

        "mse_skill_clim": {"Δ": {"vmin": -0.05, "vmax": 0.05, "ticks": 21}},
        "mae_anom_skill_clim": {"Δ": {"vmin": -0.05, "vmax": 0.05, "ticks": 21}},
        "mse_anom_skill_clim": {"Δ": {"vmin": -0.05, "vmax": 0.05, "ticks": 21}},
        "rmse_anom_skill_clim": {"Δ": {"vmin": -0.05, "vmax": 0.05, "ticks": 21}},

        "mse_bias_component": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
            "normalized": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },
        "mse_std_component": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
            "normalized": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },
        "mse_corr_component": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -2, "vmax": 2, "ticks": 11},
            "normalized": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
        },
        "crmse": {
            "%": {"vmin": -30, "vmax": 30, "ticks": 11},
            "Δ": {"vmin": -1, "vmax": 1, "ticks": 11},
            "normalized": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },

        "mse_bias_component_anom": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
            "normalized": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },
        "mse_std_component_anom": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
            "normalized": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },
        "mse_corr_component_anom": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -2, "vmax": 2, "ticks": 11},
            "normalized": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
        },
        "crmse_anom": {
            "%": {"vmin": -30, "vmax": 30, "ticks": 11},
            "Δ": {"vmin": -0.5, "vmax": 0.5, "ticks": 11},
            "normalized": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },

        "ens_member_rmse": {
            "%": {"vmin": -30, "vmax": 30, "ticks": 11},
            "Δ": {"vmin": -1, "vmax": 1, "ticks": 11},
            "normalized": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },
        "mean_member_rmse": {
            "%": {"vmin": -30, "vmax": 30, "ticks": 11},
            "Δ": {"vmin": -1, "vmax": 1, "ticks": 11},
            "normalized": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },
        "crps": {
            "%": {"vmin": -30, "vmax": 30, "ticks": 11},
            "Δ": {"vmin": -1, "vmax": 1, "ticks": 11},
            "normalized": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
        },

        "grad_rmse": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.001, "vmax": 0.001, "ticks": 13},
        },
        "grad_rmse_anom": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.001, "vmax": 0.001, "ticks": 13},
        },
    },

    # =========================================================================
    # 2 m Temperature
    # =========================================================================
    "t2m": {
        "bias": {
            "%": {"vmin": -400, "vmax": 100, "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -1, "vmax": 1, "ticks": 11},
            "normalized": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
        },
        "mae": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.6, "vmax": 0.6, "ticks": 13},
            "normalized": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
        },
        "mse": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -2, "vmax": 2, "ticks": 11},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
        "rmse": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.8, "vmax": 0.8, "ticks": 17},
            "normalized": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
        },
        "bias_anom": {
            "%": {"vmin": -400, "vmax": 100, "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -0.8, "vmax": 0.8, "ticks": 17},
            "normalized": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
        },
        "rmse_anom": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.6, "vmax": 0.6, "ticks": 13},
            "normalized": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
        },
    },

    # =========================================================================
    # Dew Point
    # =========================================================================
    "d2m": {
        "bias": {
            "%": {"vmin": -400, "vmax": 100, "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -2, "vmax": 2, "ticks": 9},
            "normalized": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
        },
        "rmse": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -1, "vmax": 1, "ticks": 9},
            "normalized": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
        },
        "rmse_anom": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.75, "vmax": 0.75, "ticks": 7},
            "normalized": {"vmin": -0.2, "vmax": 0.2, "ticks": 11},
        },
    },

    # =========================================================================
    # Wind
    # =========================================================================
    "u10": {
        "bias": {
            "%": {"vmin": -400, "vmax": 100, "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -1.5, "vmax": 1.5, "ticks": 7},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
        "rmse": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -1, "vmax": 1, "ticks": 9},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
        "rmse_anom": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.75, "vmax": 0.75, "ticks": 7},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
    },

    "v10": {
        "bias": {
            "%": {"vmin": -400, "vmax": 100, "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -1.5, "vmax": 1.5, "ticks": 7},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
        "rmse": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -1, "vmax": 1, "ticks": 9},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
        "rmse_anom": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.75, "vmax": 0.75, "ticks": 7},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
    },

    # =========================================================================
    # SST
    # =========================================================================
    "sst": {
        "bias": {
            "%": {"vmin": -400, "vmax": 100, "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -0.75, "vmax": 0.75, "ticks": 7},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
        "rmse": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.5, "vmax": 0.5, "ticks": 11},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
        "rmse_anom": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.4, "vmax": 0.4, "ticks": 9},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
    },

    # =========================================================================
    # Precipitation
    # =========================================================================
    "tprate": {
        "bias": {
            "%": {"vmin": -400, "vmax": 100, "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -2, "vmax": 2, "ticks": 9},
            "normalized": {"vmin": -0.5, "vmax": 0.5, "ticks": 11},
        },
        "rmse": {
            "%": {"vmin": -200, "vmax": 100, "ticks": [-200, -150, -100, -50, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -2, "vmax": 2, "ticks": 9},
            "normalized": {"vmin": -0.5, "vmax": 0.5, "ticks": 11},
        },
        "rmse_anom": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -1, "vmax": 1, "ticks": 9},
            "normalized": {"vmin": -0.5, "vmax": 0.5, "ticks": 11},
        },
    },

    # =========================================================================
    # Total Cloud Cover [fraction]
    # =========================================================================
    "tcc": {
        "bias": {
            "%": {"vmin": -400, "vmax": 100, "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
        "rmse": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
        "rmse_anom": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.1, "vmax": 0.1, "ticks": 11},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
    },

    # =========================================================================
    # Mixed Layer Thickness
    # =========================================================================
    "mlotst": {
        "bias": {
            "%": {"vmin": -400, "vmax": 100, "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -10, "vmax": 10, "ticks": 11},
            "normalized": {"vmin": -0.5, "vmax": 0.5, "ticks": 11},
        },
        "rmse": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -5, "vmax": 5, "ticks": 11},
            "normalized": {"vmin": -0.5, "vmax": 0.5, "ticks": 11},
        },
        "rmse_anom": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -5, "vmax": 5, "ticks": 11},
            "normalized": {"vmin": -0.5, "vmax": 0.5, "ticks": 11},
        },
    },

    # =========================================================================
    # Sea Surface Height
    # =========================================================================
    "ssh": {
        "bias": {
            "%": {"vmin": -400, "vmax": 100, "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -0.2, "vmax": 0.2, "ticks": 9},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
        "rmse": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.1, "vmax": 0.1, "ticks": 9},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
        "rmse_anom": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.1, "vmax": 0.1, "ticks": 9},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
    },

    # =========================================================================
    # Salinity
    # =========================================================================
    "sss": {
        "bias": {
            "%": {"vmin": -400, "vmax": 100, "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -1, "vmax": 1, "ticks": 9},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
        "rmse": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.5, "vmax": 0.5, "ticks": 11},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
        "rmse_anom": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.5, "vmax": 0.5, "ticks": 11},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
    },

    # =========================================================================
    # Temperature at 20 m
    # =========================================================================
    "t20d": {
        "bias": {
            "%": {"vmin": -400, "vmax": 100, "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100]},
            "Δ": {"vmin": -1.5, "vmax": 1.5, "ticks": 7},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
        "rmse": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.75, "vmax": 0.75, "ticks": 7},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
        "rmse_anom": {
            "%": {"vmin": -100, "vmax": 100, "ticks": 9},
            "Δ": {"vmin": -0.5, "vmax": 0.5, "ticks": 11},
            "normalized": {"vmin": -0.3, "vmax": 0.3, "ticks": 13},
        },
    },
}
