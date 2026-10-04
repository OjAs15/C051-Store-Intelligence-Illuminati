from src.acquisition import collect_market_profile


EXPECTED_AREAS = {
    "BKC",
    "DADAR",
    "VIKHROLI",
}


def main():
    print("Testing public market-data connector...\n")

    rows = collect_market_profile()

    print(f"Rows returned: {len(rows)}\n")

    assert len(rows) == 3, (
        f"Expected 3 rows, got {len(rows)}"
    )

    returned_areas = {
        row["area"].strip().upper()
        for row in rows
    }

    assert returned_areas == EXPECTED_AREAS, (
        f"Expected {EXPECTED_AREAS}, got {returned_areas}"
    )

    for row in rows:
        print(
            f"{row['area']}: "
            f"population={row['population']}, "
            f"purchasing_power={row['purchasing_power_proxy']}, "
            f"households={row['households']}, "
            f"status={row['status']}"
        )

    print("\nMARKET DATA CONNECTOR PASSED")


if __name__ == "__main__":
    main()
