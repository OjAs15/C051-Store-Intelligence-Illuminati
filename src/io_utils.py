from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_feature_csv(
    data: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    """
    Write standardized feature/decision records to a CSV file.

    Parameters
    ----------
    data:
        Dictionary keyed by record ID, with each value containing
        a flat dictionary of fields.

    path:
        Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(data.values())

    # Nothing to write.
    if not rows:
        return

    # Build a stable union of all columns across rows.
    fieldnames: list[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)
