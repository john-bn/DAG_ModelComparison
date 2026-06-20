# new_comparison.py
#
# Interactive front end (kept for the prompt-driven workflow documented in the
# README). The comparison engine now lives in comparator.pipeline, and the
# unattended / scriptable interface is the `compare` CLI (comparator.cli) — use
# that for cron and terminal queries. This shim just collects answers via
# input() and delegates to the same engine the CLI uses.

from datetime import datetime, timezone
import logging
import sys

from comparator import appconfig, normalize as norm, pipeline, timesel


def _setup_console_logging():
    """Surface pipeline progress (it logs instead of printing) to the console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )


def main():
    _setup_console_logging()
    cfg = appconfig.load_config()  # honors config.yaml / DAG_* env if present

    nwp_model = input(
        "Enter NWP model to compare against the analysis : "
        "HRRR, NAM5k, NAM12k, RAP, NBM, ARW, FV3, GFS, IFS, HREF: "
    ).strip()
    anl_var = input(
        "Enter analysis variable (TMP = 2m temperature, DPT = 2m dew point, "
        "VIS = visibility, WIND = 10m wind, GUST = wind gust): "
    ).strip()
    animate = input("Animate the plot? (y/n): ").strip().lower()

    # --- Validate model & variable early ---
    try:
        model_key = norm.normalize_model_key(nwp_model)
    except ValueError as e:
        print(e)
        return
    try:
        var_key = norm.normalize_var_key(anl_var)
    except ValueError as e:
        print(e)
        return

    # --- Select verification source (no default; re-prompt until valid) ---
    while True:
        verif_in = input("Verify against which analysis? (RTMA / URMA): ").strip()
        try:
            verif_key = norm.normalize_verif_key(verif_in)
            break
        except ValueError as e:
            print(e)
    verif_label = verif_key.upper()

    if animate == "y":
        # --- GIF mode: user provides the analysis time ---
        analysis_date = input(
            f"Enter the {verif_label} analysis date (YYYY-MM-DD): "
        ).strip()
        analysis_hour = int(
            input(f"Enter the {verif_label} analysis hour, in 24-hour Z-time: ")
        )
        valid_dt = datetime.fromisoformat(f"{analysis_date} {analysis_hour:02d}:00")

        gif_path = pipeline.generate_gif(
            model_key, var_key, valid_dt, verif_key,
            data_dir=cfg.data_dir, out_dir=cfg.out_dir,
        )
        if gif_path is None:
            print("No frames were generated. Cannot create GIF.")
            return
        print(f"\nGIF saved to {gif_path}")
    else:
        # --- Single-frame mode ---
        date = input("Enter date (YYYY-MM-DD): ").strip()
        init_hour = int(input("Enter a valid initialization hour, in 24-hour Z-time: "))
        forecast = int(input("Enter a valid forecast hour, in 24-hour Z-time: "))
        cycle_dt = datetime.fromisoformat(f"{date} {init_hour:02d}:00")

        result = pipeline.generate_comparison_frame(
            model_key, var_key, cycle_dt, forecast, verif_key,
            data_dir=cfg.data_dir, out_dir=cfg.out_dir,
        )
        if result is None:
            return
        print(f"Plot saved to {result.path}")


if __name__ == "__main__":
    main()
