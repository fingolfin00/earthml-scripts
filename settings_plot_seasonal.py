from earthml import (
    SeqBYRd,
)

# =============================================================================
# Colormaps and plotting configuration
# =============================================================================

VARIABLE_PLOT_CONFIG = {
    "mslp": {
        "bias": {
            "vmin": -12,
            "vmax": 12,
            "ticks": [-12, -10, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 10, 12],
        },
        "mae": {
            "vmin": 0,
            "vmax": 10,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10],
        },
        "mae_anom": {
            "vmin": 0,
            "vmax": 9,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
        "rmse": {
            "vmin": 0,
            "vmax": 10,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10],
        },
        "rmse_anom": {
            "vmin": 0,
            "vmax": 9,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
        "ens_member_rmse": {
            "vmin": 0,
            "vmax": 9,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
        "ens_member_rmse_anom": {
            "vmin": 0,
            "vmax": 9,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
        "mean_member_rmse_anom": {
            "vmin": 0,
            "vmax": 9,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
        "crps": {
            "vmin": 0,
            "vmax": 10,
            "ticks": 11,
        },
        "crps_anom": {
            "vmin": 0,
            "vmax": 9,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
        "an_std": {
            "vmin": 0,
            "vmax": 5,
            "ticks": 11,
        },
        "fc_std": {
            "vmin": 0,
            "vmax": 5,
            "ticks": 11,
        },
        "an_anom_std": {
            "vmin": 0,
            "vmax": 5,
            "ticks": 11,
        },
        "fc_anom_std": {
            "vmin": 0,
            "vmax": 5,
            "ticks": 11,
        },
    },

    "t2m": {
        "bias": {
            "vmin": -6,
            "vmax": 6,
            "ticks": [-6, -4, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 4, 6],
            "cmap": SeqBYRd,
        },
        "mae": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 4, 6],
        },
        "mae_anom": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "rmse": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "rmse_anom": {
            "vmin": 0,
            "vmax": 5,
            "ticks": [0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5],
        },
        "ens_member_rmse": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "ens_member_rmse_anom": {
            "vmin": 0,
            "vmax": 5,
            "ticks": [0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5],
        },
        "mean_member_rmse_anom": {
            "vmin": 0,
            "vmax": 5,
            "ticks": [0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5],
        },
        "crps": {
            "vmin": 0,
            "vmax": 8,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8],
        },
        "crps_anom": {
            "vmin": 0,
            "vmax": 3,
            "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3],
        },
        "an_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 11,
        },
        "fc_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 11,
        },
        "an_anom_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 11,
        },
        "fc_anom_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 11,
        },
    },

    "d2m": {
        "bias": {
            "vmin": -6,
            "vmax": 6,
            "ticks": [-6, -4, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 4, 6],
        },
        "mae": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 4, 6],
        },
        "mae_anom": {
            "vmin": 0,
            "vmax": 5,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        },
        "rmse": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "rmse_anom": {
            "vmin": 0,
            "vmax": 5,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        },
        "ens_member_rmse": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "ens_member_rmse_anom": {
             "vmin": 0,
            "vmax": 5,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        },
        "mean_member_rmse_anom": {
            "vmin": 0,
            "vmax": 5,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        },
        "crps": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 4, 6],
        },
        "crps_anom": {
            "vmin": 0,
            "vmax": 5,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        },
        "an_std": {
            "vmin": 0,
            "vmax": 12,
            "ticks": 7,
        },
        "fc_std": {
            "vmin": 0,
            "vmax": 12,
            "ticks": 7,
        },
        "an_anom_std": {
            "vmin": 0,
            "vmax": 3,
            "ticks": 7,
        },
        "fc_anom_std": {
            "vmin": 0,
            "vmax": 3,
            "ticks": 7,
        },
    },

    "u10": {
        "bias": {
            "vmin": -4,
            "vmax": 4,
            "ticks": [-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4],
        },
        "mae": {
            "vmin": 0,
            "vmax": 5,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5],
        },
        "mae_anom": {
            "vmin": 0,
            "vmax": 2,
            "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2],
        },
        "rmse": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6],
        },
        "rmse_anom": {
            "vmin": 0,
            "vmax": 2,
            "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2],
        },
        "ens_member_rmse": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6],
        },
        "ens_member_rmse_anom": {
            "vmin": 0,
            "vmax": 2,
            "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2],
        },
        "mean_member_rmse_anom": {
            "vmin": 0,
            "vmax": 2,
            "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2],
        },
        "crps": {
            "vmin": 0,
            "vmax": 5,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5],
        },
        "crps_anom": {
            "vmin": 0,
            "vmax": 2,
            "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2],
        },
        "an_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 9,
        },
        "fc_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 9,
        },
        "an_anom_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 9,
        },
        "fc_anom_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 9,
        },
    },

    "v10": {
        "bias": {
            "vmin": -4,
            "vmax": 4,
            "ticks": [-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4],
        },
        "mae": {
            "vmin": 0,
            "vmax": 5,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5],
        },
        "mae_anom": {
            "vmin": 0,
            "vmax": 2,
            "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2],
        },
        "rmse": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6],
        },
        "rmse_anom": {
            "vmin": 0,
            "vmax": 2,
            "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2],
        },
        "ens_member_rmse": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6],
        },
        "ens_member_rmse_anom": {
            "vmin": 0,
            "vmax": 2,
            "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2],
        },
        "mean_member_rmse_anom": {
            "vmin": 0,
            "vmax": 2,
            "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2],
        },
        "crps": {
            "vmin": 0,
            "vmax": 5,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5],
        },
        "crps_anom": {
            "vmin": 0,
            "vmax": 2,
            "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2],
        },
        "an_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 9,
        },
        "fc_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 9,
        },
        "an_anom_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 9,
        },
        "fc_anom_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 9,
        },
    },

    "sst": {
        "bias": {
            "vmin": -6,
            "vmax": 6,
            "ticks": [-6, -4, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 4, 6],
        },
        "mae": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 4, 6],
        },
        "mae_anom": {
            "vmin": 0,
            "vmax": 2,
            "ticks": [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
        },
        "rmse": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "rmse_anom": {
            "vmin": 0,
            "vmax": 3,
            "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3],
        },
        "ens_member_rmse": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "ens_member_rmse_anom": {
            "vmin": 0,
            "vmax": 3,
            "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3],
        },
        "mean_member_rmse_anom": {
            "vmin": 0,
            "vmax": 3,
            "ticks": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3],
        },
        "crps": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "crps_anom": {
             "vmin": 0,
            "vmax": 1,
            "ticks": 7,
        },
        "an_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 11,
        },
        "fc_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 11,
        },
        "an_anom_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 11,
        },
        "fc_anom_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 11,
        },
    },

    "tprate": { # mm/day
        "bias": {
            "vmin": -10,
            "vmax": 10,
            "ticks": [-10, -8, -6, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 6, 8, 10],
        },
        "mae": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 4, 6],
        },
        "mae_anom": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "rmse": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "rmse_anom": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "ens_member_rmse": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "ens_member_rmse_anom": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "mean_member_rmse_anom": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "crps": {
            "vmin": 0,
            "vmax": 10,
            "ticks": 11,
        },
        "crps_anom": {
             "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "an_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 11,
        },
        "fc_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 11,
        },
        "an_anom_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 11,
        },
        "fc_anom_std": {
            "vmin": 0,
            "vmax": 2,
            "ticks": 11,
        },
    },
}

IMPROVEMENT_PLOT_CONFIG = {
    "mslp": {
        "bias": {
            "vmin": -400,
            "vmax": 100,
            "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100],
        },
        "rmse": {
            "vmin": -200,
            "vmax": 100,
            "ticks": [-200, -150, -100, -50, 0, 25, 50, 75, 100],
        },
        "rmse_anom": {
            "vmin": -100,
            "vmax": 100,
            "ticks": [-100, -75, -50, -25, 0, 25, 50, 75, 100],
        },
    },
    "t2m": {
        "bias": {
            "vmin": -400,
            "vmax": 100,
            "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100],
        },
        "rmse": {
            "vmin": -100,
            "vmax": 100,
            "ticks": [-100, -75, -50, -25, 0, 25, 50, 75, 100],
        },
        "rmse_anom": {
            "vmin": -100,
            "vmax": 100,
            "ticks": [-100, -75, -50, -25, 0, 25, 50, 75, 100],
        },
    },
}
