"""Score saved forecasts against what actually happened.

    python -m smartgrid.score

Only measures days that have already passed and been ingested. A forecast saved
today can first be scored tomorrow, once generation for it exists.
"""

import pandas as pd

from smartgrid.modelling.store import score_predictions, summarise_scores


def main(argv: list[str] | None = None) -> int:
    pd.set_option("display.width", 200)

    scored = score_predictions()
    if scored.empty:
        print(
            "No saved forecast has a measured outcome yet.\n"
            "Save one with: python -m smartgrid.tomorrow --save\n"
            "then score it after the day has passed and been ingested."
        )
        return 0

    summary = summarise_scores(scored)
    print(f"{len(scored):,} forecast hours with a measured outcome\n")
    print(summary.round(4).to_string())

    daylight = scored[scored["ghi_mean_forecast"] > 0].copy()
    daylight["error"] = (
        daylight["predicted_capacity_factor"] - daylight["actual_capacity_factor"]
    )

    print("\nby target day:")
    by_day = daylight.groupby("target_date").agg(
        hours=("error", "size"),
        mae=("error", lambda e: e.abs().mean()),
        bias=("error", "mean"),
    )
    print(by_day.round(4).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
