from pathlib import Path
from collections import defaultdict

import numpy as np
import xarray as xr

import warnings
from dask.array import PerformanceWarning

warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=PerformanceWarning,
)

from dask.diagnostics.progress import ProgressBar

import earthml
from earthml import (
    LeadtimeUnit,
    ClimPeriod,
    get_experiment_configs,
    get_and_subset_datasets,
)
from earthml.metrics import (
    LeadtimeAgg,
    MetricAgg,
    is_deterministic,
    is_probabilistic,
    get_metrics,
    calculate_save_and_subset_climatologies,
    stack_hour_clim,
    groupby_period,
)
from earthml.plots import (
    safe_label,
    PlotMode,
    plot_profile,
)


def get_profile_metrics(
    *,
    s,
    an,
    fc,
    an_clim,
    fc_clim,
    metrics,
    realization_agg,
    metric_agg_mode,
    leadtime_agg_mode,
    leadtime_agg_coord,
    clim_period,
    period_dim,
    wanted_start_periods,
):
    metric_kind = (
        "scalar"
        if metric_agg_mode == "global"
        else "maps"
    )

    ds = get_metrics(
        an=an,
        fc=fc,
        var=s.var_fc,
        metric_kind=metric_kind,
        leadtime_agg=leadtime_agg_mode,
        realization_agg=realization_agg,
        an_clim=an_clim,
        fc_clim=fc_clim,
        metrics=metrics,
        leadtime_windows=s.seasonal_leadtime_windows,
        leadtime_agg_coord=leadtime_agg_coord,
        clim_period=clim_period,
        period_dim=period_dim,
        periods_requested=wanted_start_periods,
    )

    if metric_agg_mode == "global":
        return ds

    if metric_agg_mode == "spatial_avg":
        lat_dim = fc.earthml.guessed_dims.latitude
        lon_dim = fc.earthml.guessed_dims.longitude

        weights = np.cos(np.deg2rad(fc[lat_dim]))

        return ds.weighted(weights).mean(
            dim=(lat_dim, lon_dim)
        )

    raise ValueError(
        f"Unsupported metric_agg_mode={metric_agg_mode}"
    )


def difference_from_reference(
    das_by_model: dict[str, xr.Dataset],
    reference: str,
    mode: str = "model_minus_reference",
) -> dict[str, xr.Dataset]:
    if reference not in das_by_model:
        raise ValueError(
            f"Difference reference {reference!r} is not available. "
            f"Available models: {list(das_by_model)}"
        )

    if mode not in {
        "model_minus_reference",
        "reference_minus_model",
    }:
        raise ValueError(
            f"Unsupported difference_mode={mode!r}. "
            "Choose 'model_minus_reference' or "
            "'reference_minus_model'."
        )

    reference_ds = das_by_model[reference]

    diff_by_model: dict[str, xr.Dataset] = {}

    for model, ds in das_by_model.items():
        # Do not plot reference-reference = 0.
        if model == reference:
            continue

        reference_aligned, ds_aligned = xr.align(
            reference_ds,
            ds,
            join="exact",
        )

        if mode == "model_minus_reference":
            diff = ds_aligned - reference_aligned
        else:
            diff = reference_aligned - ds_aligned

        diff_by_model[model] = diff

    return diff_by_model


def main() -> None:
    experiments_root = Path("/Users/jacopodallaglio/ML/training/seasonal/experiments")

    # Shared plot directory, same idea used by the map plotting script.
    fc_plot_dir = Path(
        "/Users/jacopodallaglio/ML/training/seasonal/plots"
    )

    plot_mode: PlotMode = "profiles"

    # --------------------------------------------------------------
    # Plot modes
    # --------------------------------------------------------------
    # Preserve previous behaviour: each experiment gets its own
    # fc/mlfc profile plots.
    plot_individual_experiments = False

    # New behaviour: selected experiments can additionally be shown
    # together in common comparison plots.
    plot_combined_experiments = True

    # Folder under the common plot directory.
    combined_plot_folder = "profile_comparison"
    comparison_name = "smaatunet-convnexttransformer"

    comparison_labels = [
        "ConvNeXt reanalysis",
        "ConvNeXt reanalysis ens mean",
        "SmaAt-UNet anomaly residual ens mean",
        "SmaAt-UNet anomaly residual",
    ]

    comparison_colors = {
        "fc": "tab:blue",
        "clim-fc": "tab:orange",
        "ConvNeXt reanalysis": "green",
        "ConvNeXt reanalysis ens mean": "green",
        "SmaAt-UNet anomaly residual ens mean": "red",
        "SmaAt-UNet anomaly residual": "red",
    }

    comparison_linestyles = {
        "fc": "-",
        "clim-fc": "-",
        "ConvNeXt reanalysis": "-",
        "ConvNeXt reanalysis ens mean": "--",
        "SmaAt-UNet anomaly residual ens mean": "--",
        "SmaAt-UNet anomaly residual": "-",
    }

    regenerate_plots = False

    include_clim_fc = False

    # Plot metric differences instead of absolute metric values.
    plot_difference = True

    # Any model available in the combined plot:
    # "fc", "clim-fc", or one of comparison_labels.
    difference_reference = "fc"

    # Difference convention:
    #     "model_minus_reference" -> model - reference
    #     "reference_minus_model" -> reference - model
    difference_mode = "model_minus_reference"

    force_clim_recalc = False
    interpolate = True
    build_analysis = True
    materialize_once = False

    metrics = [
        # ==========================================================
        # Deterministic Metrics (Absolute Fields)
        # ==========================================================
        "bias",
        # "mae",
        # "mse",
        "rmse",
        # "nrmse",
        # "corr",
        # "r2",
        # "fc_std",
        # "an_std",
        # "std_ratio",

        # ==========================================================
        # Deterministic Metrics (Anomaly Fields)
        # ==========================================================
        # "bias_anom",
        # "mae_anom",
        # "mse_anom",
        # "rmse_anom",
        # "nrmse_anom",
        # "acc",
        # "r2_anom",
        # "fc_anom_std",
        # "an_anom_std",
        # "std_ratio_anom",

        # ==========================================================
        # Skill Scores vs Climatology
        # ==========================================================
        # "mse_skill_clim",
        # "mae_anom_skill_clim",
        # "mse_anom_skill_clim",
        # "rmse_anom_skill_clim",
        # "ens_member_mse_anom_skill_clim",
        # "mean_member_mse_anom_skill_clim",

        # ==========================================================
        # Ensemble / Probabilistic Metrics (Absolute Fields)
        # ==========================================================
        # "ens_member_rmse",
        # "mean_member_rmse",
        # "spread",
        # "spread_skill_ratio",
        # "crps",
        # "rank_histogram",

        # ==========================================================
        # Ensemble / Probabilistic Metrics (Anomaly Fields)
        # ==========================================================
        # "ens_member_rmse_anom",
        # "mean_member_rmse_anom",
        # "spread_anom",
        # "spread_anom_skill_ratio",
        # "crps_anom",
        # "rank_histogram_anom",

        # ==========================================================
        # ROC AUC (Anomaly Terciles)
        # ==========================================================
        # "roc_anom_lower",
        # "roc_anom_middle",
        # "roc_anom_upper",
    ]

    variables = [
        # Atmo
        "mslp",
        # "t2m",
        # "d2m",
        # "u10",
        # "v10",
        # "sst",
        # "tprate",
        # "tcc",
        # Ocean
        # "mlotst",
        # "ssh",
        # "sss",
        # "t20d",
    ]
    regions = [
        # "ConUS",
        # "Europe",
        # "Pacific",
        "World",
        # None, # accept all
    ]

    # ConUS
    # lat_range = (50, 25)
    # lon_range = (-130, -60)
    # Europe
    # lat_range = (80, 30)
    # lon_range = (-30, 60)
    # Pacific
    # lat_range = (20, -20)
    # lon_range = (-195, -135)
    # World or whole region
    lat_range = None
    lon_range = None

    wanted_start_periods = [
        "01",
        "05",
        "08",
        "10",
        "all",
    ]

    leadtime_units = LeadtimeUnit.MONTHS
    clim_period: ClimPeriod = ClimPeriod.MONTH # "dayofyear", "day", "month", "year", "day_hour", "dayofyear_hour", "month_hour"
    clim_rolling_window = None
    # clim_period: ClimPeriod = ClimPeriod.DAYOFYEAR_HOUR # "dayofyear", "day", "month", "year", "day_hour", "dayofyear_hour", "month_hour"
    # clim_rolling_window = 31

    time_range = None
    # time_range = ("2018-01-01", "2022-12-31")

    inference_period = None
    # inference_period = ("2025-01-01", "2025-10-31")

    metric_agg_mode: MetricAgg = "spatial_avg" # "spatial_avg", "global", "spatial_rmse"
    leadtime_agg_mode: LeadtimeAgg = "single" # "single", "aggregated", "seasonal_window"
    plot_members = True

    settings = get_experiment_configs(
        experiments_root,
        var_fc=variables,
        region_name=regions,
        # net_name="ConvNeXtTransformerUNet",
        # net_name="SmaAt_UNet",
        # target_mode="anomaly",
        # seasonal_encoding=True,
        # ensemble_encoding=True,
        # input_realization_avg=True,
        # channel_representation="variable",
        # loss_name="VarNormMaskMSELoss",
        # loss_name="GeoMaskedMSEMultiScaleLoss",
        # loss_name="SpatialDegradationMSELoss",
        # separate_training_by_init_period=None,
        # separate_training_by_init_period=ClimPeriod.MONTH,
        # pretrain_norm="full",
        # extra_suffix_folder="inputs-mslp-u10-v10",
        # extra_suffix_folder="",
    )

    print(f"Found {len(settings)} matching experiment(s).")

    # ==============================================================
    # Combined-experiment storage
    #
    # Keyed by the properties that must be common for a meaningful
    # comparison. Different variables/regions/time ranges therefore
    # automatically go to independent comparison groups.
    # ==============================================================

    combined_groups = defaultdict(
        lambda: {
            "fc": None,
            "fc_members": None,
            "clim-fc": None,
            "clim-fc_members": None,
            "mlfc": [],
            "mlfc_members": [],
            "labels": [],
            "settings": [],
            "valid_time_range": None,
            "valid_lat_range": None,
            "valid_lon_range": None,
        }
    )

    n = 0

    for s in settings:
        if inference_period is None:
            valid_time_range = (
                (s.test_start, s.test_end)
                # (s.train_start, s.train_end)
                if time_range is None
                else time_range
            )
            mlfc_path = None

        else:
            valid_time_range = (
                inference_period
                if time_range is None
                else time_range
            )

            inference_start, inference_end = inference_period

            mlfc_path = (
                s.exp_dir
                / "inference"
                / f"{inference_start}_{inference_end}"
                / "test_corrected.zarr"
            )

        # clim_time_range = (s.train_start, s.val_end)
        clim_time_range = (s.train_start, s.train_end)

        lat_lon = (
            list(s.region.values())
            if s.region is not None
            else [None, None]
        )

        valid_lat_range = (
            lat_lon[0]
            if lat_range is None
            else lat_range
        )

        valid_lon_range = (
            lat_lon[1]
            if lon_range is None
            else lon_range
        )

        leadtime_agg_coord = (
            "leadtime"
            if leadtime_agg_mode == "single"
            else "leadtime_seasonal"
        )

        print(
            f"Generate {leadtime_agg_mode} {plot_mode} "
            f"for {(s.var_an, s.var_fc)} in {s.region_name} "
            f"(lon={valid_lon_range}, lat={valid_lat_range})"
        )

        fc, an, mlfc = get_and_subset_datasets(
            s,
            leadtime_units=leadtime_units,
            lat_range=valid_lat_range,
            lon_range=valid_lon_range,
            time_range=valid_time_range,
            interpolate=interpolate,
            mlfc_path=mlfc_path,
        )

        if mlfc is not None:
            mlfc = mlfc.assign_coords(
                leadtime=s.leadtimes
            )

        fc_clim, an_clim, mlfc_clim = (
            calculate_save_and_subset_climatologies(
                s,
                leadtime_units=leadtime_units,
                force=force_clim_recalc,
                clim_period=clim_period,
                rolling_window=clim_rolling_window,
                rolling_center=True,
                rolling_min_periods=1,
                lat_range=valid_lat_range,
                lon_range=valid_lon_range,
                time_range=clim_time_range,
                time_start=None,
                interpolate=interpolate,
                engine="zarr",
                build_analysis=build_analysis,
                coord_rename_fc=None,
                coord_rename_an=None,
            )
        )

        if mlfc_clim is not None:
            mlfc_clim = mlfc_clim.assign_coords(
                leadtime=s.leadtimes
            )

        leadtime_dim = fc.earthml.guessed_dims.leadtime

        fc = fc.sel({leadtime_dim: s.leadtimes})
        an = an.sel({leadtime_dim: s.leadtimes})
        fc_clim = fc_clim.sel({leadtime_dim: s.leadtimes})
        an_clim = an_clim.sel({leadtime_dim: s.leadtimes})

        # ----------------------------------------------------------
        # Forecast corrected only by replacing forecast climatology
        # with analysis climatology.
        # ----------------------------------------------------------

        fc_clim_da = stack_hour_clim(
            fc_clim[s.var_fc],
            clim_period,
        )

        an_clim_da = stack_hour_clim(
            an_clim[s.var_an],
            clim_period,
        )

        fc_anom_da = (
            groupby_period(
                fc[s.var_fc],
                fc.earthml.guessed_dims.time,
                clim_period,
            )
            - fc_clim_da
        )

        clim_fc = (
            groupby_period(
                fc_anom_da,
                fc.earthml.guessed_dims.time,
                clim_period,
            )
            + an_clim_da
        ).to_dataset(name=s.var_fc)

        # Analysis climatology is the appropriate forecast
        # climatology for clim-fc.
        an_clim_for_fc = an_clim

        realization_dim = fc.earthml.guessed_dims.realization

        if (
            realization_dim is not None
            and realization_dim in fc.dims
            and realization_dim not in an_clim_for_fc.dims
        ):
            an_clim_for_fc = an_clim_for_fc.expand_dims(
                {
                    realization_dim:
                        fc[realization_dim]
                }
            )

        datasets = {
            "fc": (fc, fc_clim),
            "mlfc": (mlfc, mlfc_clim),
        }

        if include_clim_fc:
            datasets["clim-fc"] = (
                clim_fc,
                an_clim_for_fc,
            )

        if plot_mode not in {"profiles", "all"}:
            continue

        deterministic_metrics = [
            m
            for m in metrics
            if is_deterministic(m)
        ]

        probabilistic_metrics = [
            m
            for m in metrics
            if is_probabilistic(m)
        ]

        metrics_by_model: dict[str, xr.Dataset] = {}
        members_by_model: dict[str, xr.Dataset] = {}

        for model, (ds, ds_clim) in datasets.items():
            if ds is None or ds_clim is None:
                continue

            print(f"Get {model} profile metrics")

            metric_parts = []

            # ------------------------------------------------------
            # Deterministic / ensemble-mean metrics
            # ------------------------------------------------------

            if deterministic_metrics:
                metric_parts.append(
                    get_profile_metrics(
                        s=s,
                        an=an,
                        fc=ds,
                        an_clim=an_clim,
                        fc_clim=ds_clim,
                        metrics=deterministic_metrics,
                        realization_agg=True,
                        metric_agg_mode=metric_agg_mode,
                        leadtime_agg_mode=leadtime_agg_mode,
                        leadtime_agg_coord=leadtime_agg_coord,
                        clim_period=clim_period,
                        period_dim=f"start_{leadtime_units}",
                        wanted_start_periods=wanted_start_periods,
                    )
                )

            # ------------------------------------------------------
            # Probabilistic metrics require realizations
            # ------------------------------------------------------

            if probabilistic_metrics:
                metric_parts.append(
                    get_profile_metrics(
                        s=s,
                        an=an,
                        fc=ds,
                        an_clim=an_clim,
                        fc_clim=ds_clim,
                        metrics=probabilistic_metrics,
                        realization_agg=False,
                        metric_agg_mode=metric_agg_mode,
                        leadtime_agg_mode=leadtime_agg_mode,
                        leadtime_agg_coord=leadtime_agg_coord,
                        clim_period=clim_period,
                        period_dim=f"start_{leadtime_units}",
                        wanted_start_periods=wanted_start_periods,
                    )
                )

            if metric_parts:
                metrics_by_model[model] = xr.merge(
                    metric_parts
                )

            # ------------------------------------------------------
            # Individual-member deterministic metrics
            # ------------------------------------------------------

            if plot_members and deterministic_metrics:
                members_by_model[model] = (
                    get_profile_metrics(
                        s=s,
                        an=an,
                        fc=ds,
                        an_clim=an_clim,
                        fc_clim=ds_clim,
                        metrics=deterministic_metrics,
                        realization_agg=False,
                        metric_agg_mode=metric_agg_mode,
                        leadtime_agg_mode=leadtime_agg_mode,
                        leadtime_agg_coord=leadtime_agg_coord,
                        clim_period=clim_period,
                        period_dim=f"start_{leadtime_units}",
                        wanted_start_periods=wanted_start_periods,
                    )
                )

        if not metrics_by_model:
            print(
                f"No metrics available for {s.output_name}"
            )
            continue

        if materialize_once:
            with ProgressBar():
                metrics_by_model = {
                    model: ds.compute()
                    for model, ds
                    in metrics_by_model.items()
                }

                members_by_model = {
                    model: ds.compute()
                    for model, ds
                    in members_by_model.items()
                }

        # ==========================================================
        # Previous functionality:
        # individual profile plots for each experiment
        # ==========================================================

        if plot_individual_experiments:
            available_models = tuple(
                model
                for model in datasets.keys()
                if model in metrics_by_model
            )

            plot_metrics_by_model = metrics_by_model
            plot_members_by_model = members_by_model

            if plot_difference:
                plot_metrics_by_model = difference_from_reference(
                    metrics_by_model,
                    reference=difference_reference,
                    mode=difference_mode,
                )

                # Disable member profiles in diff mode for now.
                plot_members_by_model = {}

                available_models = tuple(
                    plot_metrics_by_model.keys()
                )

            available_metrics = [
                metric
                for metric in metrics
                if metric != "rank_histogram"
                and all(
                    metric in plot_metrics_by_model[model].data_vars
                    for model in available_models
                )
            ]

            reference_ds = plot_metrics_by_model[
                available_models[0]
            ]

            start_periods = [
                str(x)
                for x in reference_ds[
                    f"start_{leadtime_units}"
                ].values
                if str(x) in wanted_start_periods
            ]

            print(
                f"Plotting metric profiles "
                f"{available_metrics} "
                f"for periods {start_periods} "
                f"for exp {s.output_name}"
            )

            for metric in available_metrics:
                das = [
                    plot_metrics_by_model[model][metric]
                    for model in available_models
                ]

                das_member = []

                for model in available_models:
                    member_ds = plot_members_by_model.get(model)

                    if (
                        member_ds is not None
                        and metric
                        in member_ds.data_vars
                    ):
                        das_member.append(
                            member_ds[metric]
                        )
                    else:
                        das_member.append(None)

                plot_variant = (
                    f"diff_vs_{safe_label(difference_reference)}"
                    if plot_difference
                    else "absolute"
                )

                for start_period in start_periods:
                    common_path = (
                        Path("profiles")
                        / plot_variant
                        / safe_label(start_period)
                        / (
                            f"time_{safe_label(valid_time_range)}"
                            f"_lat_{safe_label(valid_lat_range)}"
                            f"_lon_{safe_label(valid_lon_range)}"
                        )
                        / metric
                        / metric_agg_mode
                    )

                    filename = (
                        f"{s.var_fc}_{metric}_"
                        f"{leadtime_agg_mode}lt"
                        + (
                            f"_diff_vs_{safe_label(difference_reference)}"
                            if plot_difference
                            else ""
                        )
                        + ".png"
                    )

                    out_file = (
                        s.plot_dir
                        / common_path
                        / filename
                    )

                    if (
                        out_file.exists()
                        and not regenerate_plots
                    ):
                        continue

                    print(
                        f"Saving profile {out_file}"
                    )

                    plot_profile(
                        das=das,
                        var=s.var_fc,
                        metric=metric,
                        start_period=start_period,
                        models=available_models,
                        out_file=out_file,
                        time_range=valid_time_range,
                        das_member=das_member,
                        leadtime_dim=leadtime_agg_coord,
                        leadtime_unit=leadtime_units,
                        period_dim=f"start_{leadtime_units}",
                        realization_dim="realization",
                        spread="std",
                        plot_single_members=plot_members,
                        difference_reference=(
                            difference_reference
                            if plot_difference
                            else None
                        ),
                        difference_mode=difference_mode,
                    )

                    n += 1

        # ==========================================================
        # New functionality:
        # accumulate experiments for combined plots
        # ==========================================================

        if (
            plot_combined_experiments
            and "mlfc" in metrics_by_model
        ):
            comparison_key = (
                s.var_fc,
                s.region_name,
                safe_label(valid_time_range),
                safe_label(valid_lat_range),
                safe_label(valid_lon_range),
                tuple(s.leadtimes),
                leadtime_agg_mode,
                metric_agg_mode,
            )

            group = combined_groups[comparison_key]

            group["valid_time_range"] = (
                valid_time_range
            )
            group["valid_lat_range"] = (
                valid_lat_range
            )
            group["valid_lon_range"] = (
                valid_lon_range
            )

            # FC is common to all selected experiments in this
            # comparison group and is therefore stored only once.
            if (
                group["fc"] is None
                and "fc" in metrics_by_model
            ):
                group["fc"] = (
                    metrics_by_model["fc"]
                )

                group["fc_members"] = (
                    members_by_model.get("fc")
                )

            if (
                include_clim_fc
                and group["clim-fc"] is None
                and "clim-fc" in metrics_by_model
            ):
                group["clim-fc"] = metrics_by_model["clim-fc"]
                group["clim-fc_members"] = members_by_model.get("clim-fc")

            group["mlfc"].append(
                metrics_by_model["mlfc"]
            )

            group["mlfc_members"].append(
                members_by_model.get("mlfc")
            )

            # output_name uniquely identifies the experiment and
            # keeps the mapping between curve and experiment clear.
            group["labels"].append(
                s.output_name
            )

            group["settings"].append(s)

    # ==============================================================
    # Combined experiment profile plots
    # ==============================================================

    if plot_combined_experiments:
        for comparison_key, group in (
            combined_groups.items()
        ):
            if not group["mlfc"]:
                continue

            common_s = group["settings"][0]

            valid_time_range = (
                group["valid_time_range"]
            )
            valid_lat_range = (
                group["valid_lat_range"]
            )
            valid_lon_range = (
                group["valid_lon_range"]
            )

            # ------------------------------------------------------
            # Models:
            #
            #     fc
            #     experiment_1
            #     experiment_2
            #     ...
            #
            # FC is intentionally plotted only once.
            # ------------------------------------------------------

            das_by_model = {}
            members_by_combined_model = {}

            if group["fc"] is not None:
                das_by_model["fc"] = group["fc"]

                if group["fc_members"] is not None:
                    members_by_combined_model["fc"] = group["fc_members"]

            if (
                include_clim_fc
                and group["clim-fc"] is not None
            ):
                das_by_model["clim-fc"] = group["clim-fc"]

                if group["clim-fc_members"] is not None:
                    members_by_combined_model["clim-fc"] = group["clim-fc_members"]

            if len(comparison_labels) != len(group["mlfc"]):
                raise ValueError(
                    "comparison_labels must contain one label for each "
                    f"combined experiment. Got {len(comparison_labels)} labels "
                    f"for {len(group['mlfc'])} experiments."
                )

            for (
                experiment_label,
                mlfc_metrics,
                mlfc_members,
            ) in zip(
                comparison_labels,
                group["mlfc"],
                group["mlfc_members"],
                strict=True,
            ):
                das_by_model[experiment_label] = mlfc_metrics

                if mlfc_members is not None:
                    members_by_combined_model[
                        experiment_label
                    ] = mlfc_members

            # --------------------------------------------------------------
            # Optional difference against a selectable reference model
            # --------------------------------------------------------------

            if plot_difference:
                das_by_model = difference_from_reference(
                    das_by_model,
                    reference=difference_reference,
                    mode=difference_mode,
                )

                # Member differences are deliberately disabled here.
                # The reference profile is an aggregated metric, while member
                # metrics have an additional realization dimension and should
                # not be mixed implicitly.
                members_by_combined_model = {}

            available_models = tuple(
                das_by_model.keys()
            )

            model_colors = comparison_colors
            model_linestyles = comparison_linestyles

            available_metrics = [
                metric
                for metric in metrics
                if metric != "rank_histogram"
                and all(
                    metric
                    in das_by_model[
                        model
                    ].data_vars
                    for model in available_models
                )
            ]

            reference_ds = (
                das_by_model[
                    available_models[0]
                ]
            )

            start_periods = [
                str(x)
                for x in reference_ds[
                    f"start_{leadtime_units}"
                ].values
                if str(x) in wanted_start_periods
            ]

            print(
                "Plotting combined experiment "
                f"profiles {available_metrics} "
                f"for periods {start_periods}: "
                f"{group['labels']}"
            )

            # ------------------------------------------------------
            # Folder name identifying exactly which experiments
            # participate in the comparison.
            # ------------------------------------------------------

            for metric in available_metrics:
                das = [
                    das_by_model[model][metric]
                    for model in available_models
                ]

                das_member = []

                for model in available_models:
                    member_ds = (
                        members_by_combined_model.get(
                            model
                        )
                    )

                    if (
                        member_ds is not None
                        and metric
                        in member_ds.data_vars
                    ):
                        das_member.append(
                            member_ds[metric]
                        )
                    else:
                        das_member.append(None)

                for start_period in start_periods:
                    plot_variant = (
                        f"diff_vs_{safe_label(difference_reference)}"
                        if plot_difference
                        else "absolute"
                    )

                    common_path = (
                        Path("profiles")
                        / combined_plot_folder
                        / comparison_name
                        / plot_variant
                        / safe_label(start_period)
                        / (
                            f"time_"
                            f"{safe_label(valid_time_range)}"
                            f"_lat_"
                            f"{safe_label(valid_lat_range)}"
                            f"_lon_"
                            f"{safe_label(valid_lon_range)}"
                        )
                        / metric
                        / metric_agg_mode
                    )

                    filename = (
                        f"{common_s.var_fc}_"
                        f"{metric}_"
                        f"{leadtime_agg_mode}lt"
                        + (
                            f"_diff_vs_{safe_label(difference_reference)}"
                            if plot_difference
                            else ""
                        )
                        + ".png"
                    )

                    out_file = (
                        fc_plot_dir
                        / common_path
                        / filename
                    )

                    if (
                        out_file.exists()
                        and not regenerate_plots
                    ):
                        continue

                    print(
                        "Saving combined profile "
                        f"{out_file}"
                    )

                    plot_profile(
                        das=das,
                        var=common_s.var_fc,
                        metric=metric,
                        start_period=start_period,
                        models=available_models,
                        out_file=out_file,
                        time_range=valid_time_range,
                        das_member=das_member,
                        leadtime_dim=leadtime_agg_coord,
                        leadtime_unit=leadtime_units,
                        period_dim=f"start_{leadtime_units}",
                        realization_dim="realization",
                        spread="std",
                        plot_single_members=plot_members,
                        model_colors=model_colors,
                        model_linestyles=model_linestyles,
                        difference_reference=(
                            difference_reference
                            if plot_difference
                            else None
                        ),
                        difference_mode=difference_mode,
                    )

                    n += 1

    print(f"Done. Saved {n} plots.")


if __name__ == "__main__":
    main()
