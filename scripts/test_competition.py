from src.acquisition import collect_competition


EXPECTED_AREAS = {
    "BKC",
    "DADAR",
    "VIKHROLI",
}


def main():
    print(
        "Testing real competition connector...\n"
    )

    rows = collect_competition()

    print(
        f"Rows returned: {len(rows)}\n"
    )

    # --------------------------------------------------------
    # Every candidate MUST be represented.
    # --------------------------------------------------------

    returned_areas = {
        row["area"].strip().upper()
        for row in rows
        if row.get("area")
    }

    missing_areas = (
        EXPECTED_AREAS - returned_areas
    )

    assert not missing_areas, (
        "Missing candidate areas: "
        f"{missing_areas}"
    )

    # --------------------------------------------------------
    # Display results candidate by candidate.
    # --------------------------------------------------------

    for area in sorted(
        EXPECTED_AREAS
    ):

        area_rows = [
            row
            for row in rows
            if (
                row.get("area", "")
                .strip()
                .upper()
                == area
            )
        ]

        valid_competitors = [
            row
            for row in area_rows
            if row.get("status") == "ok"
        ]

        error_rows = [
            row
            for row in area_rows
            if row.get("status") == "error"
        ]

        no_match_rows = [
            row
            for row in area_rows
            if row.get("status") == "no_match"
        ]

        print(
            f"{area}: "
            f"{len(valid_competitors)} "
            f"competitor records"
        )

        for row in valid_competitors[:10]:

            print(
                f"    "
                f"{row['competitor_name']} | "
                f"{row['category']} | "
                f"{row['distance_km']} km"
            )

        if no_match_rows:
            print(
                "    No relevant named competitor "
                "records returned."
            )

        if error_rows:
            print(
                "    WARNING: source error occurred."
            )

            for row in error_rows:
                print(
                    f"    {row.get('error')}"
                )

        print()

    # --------------------------------------------------------
    # Structural checks
    # --------------------------------------------------------

    for row in rows:

        assert row.get(
            "area"
        ), "Every row must have an area."

        assert row.get(
            "area_id"
        ), "Every row must have an area_id."

        assert row.get(
            "status"
        ) in {
            "ok",
            "no_match",
            "error",
        }, (
            "Unexpected competition status: "
            f"{row.get('status')}"
        )

        if row["status"] == "ok":

            assert (
                row["competitor_name"]
            ), (
                "A valid competitor record "
                "must have a name."
            )

            assert (
                row["distance_km"]
                is not None
            ), (
                "A valid competitor record "
                "must have distance."
            )

            assert (
                row["distance_km"]
                <= 2.0
            ), (
                "Competitor lies outside "
                "the 2 km catchment."
            )

    print(
        "COMPETITION CONNECTOR PASSED"
    )


if __name__ == "__main__":
    main()
