from src.acquisition import collect_accessibility


EXPECTED_AREAS = {
    "BKC",
    "DADAR",
    "VIKHROLI",
}


def main():

    print("Testing real accessibility connector...\n")

    rows = collect_accessibility()

    print(f"Rows returned: {len(rows)}\n")

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

    for row in rows:

        print(
            f"{row['area']}: "
            f"transit_nodes={row['transit_nodes']}, "
            f"within_500m={row['transit_nodes_500m']}, "
            f"within_1km={row['transit_nodes_1km']}, "
            f"within_2km={row['transit_nodes_2km']}, "
            f"nearest={row['nearest_transit_distance_km']} km, "
            f"average={row['average_transit_distance_km']} km, "
            f"status={row['status']}"
        )

    print("\nACCESSIBILITY CONNECTOR PASSED")


if __name__ == "__main__":
    main()
