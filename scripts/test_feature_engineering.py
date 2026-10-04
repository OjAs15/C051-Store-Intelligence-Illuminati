from src.features import (
    build_candidate_features,
    write_feature_csv,
)


EXPECTED_AREAS = {
    "BKC",
    "DADAR",
    "VIKHROLI",
}


REQUIRED_COLUMNS = {
    "area_id",
    "area",
    "latitude",
    "longitude",

    "population",
    "population_density_per_sqkm",
    "households",
    "household_density_per_sqkm",
    "purchasing_power_proxy",

    "competitor_count",
    "competitor_pressure",
    "nearest_competitor_distance_km",

    "transit_nodes",
    "transit_nodes_500m",
    "transit_nodes_1km",
    "transit_nodes_2km",
    "nearest_transit_distance_km",
    "transit_density_per_sqkm",

    "monthly_rent",
    "annual_rent",

    "catchment_radius_km",
    "catchment_area_sqkm",
}


def main():

    print("Testing feature engineering...\n")

    rows = build_candidate_features()

    print(
        f"Feature rows generated: {len(rows)}\n"
    )

    assert len(rows) == 3, (
        f"Expected 3 candidate rows, got {len(rows)}"
    )

    returned_areas = {
        row["area"].strip().upper()
        for row in rows
    }

    assert returned_areas == EXPECTED_AREAS, (
        f"Expected {EXPECTED_AREAS}, "
        f"got {returned_areas}"
    )

    actual_columns = set(rows[0].keys())

    missing_columns = (
        REQUIRED_COLUMNS - actual_columns
    )

    assert not missing_columns, (
        f"Missing feature columns: {missing_columns}"
    )

    for row in rows:

        assert row["latitude"] is not None
        assert row["longitude"] is not None

        assert row["catchment_radius_km"] > 0
        assert row["catchment_area_sqkm"] > 0

        assert row["competitor_count"] >= 0
        assert row["competitor_pressure"] >= 0

        assert row["transit_nodes"] >= 0
        assert row["transit_nodes_500m"] >= 0
        assert row["transit_nodes_1km"] >= 0
        assert row["transit_nodes_2km"] >= 0

        if row["nearest_transit_distance_km"] is not None:
            assert (
                row["nearest_transit_distance_km"]
                >= 0
            )

        if row["nearest_competitor_distance_km"] is not None:
            assert (
                row["nearest_competitor_distance_km"]
                >= 0
            )

        print(
            f"{row['area']}: "
            f"population={row['population']}, "
            f"competitors={row['competitor_count']}, "
            f"competition_pressure={row['competitor_pressure']}, "
            f"transit_nodes={row['transit_nodes']}, "
            f"nearest_transit="
            f"{row['nearest_transit_distance_km']} km"
        )

    output_path = write_feature_csv(rows)

    print(
        f"\nFeature CSV written to:\n"
        f"{output_path}"
    )

    print("\nFEATURE ENGINEERING PASSED")


if __name__ == "__main__":
    main()
