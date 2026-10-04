from src.acquisition import collect_public_data


def main():
    print("Running acquisition scaffold check...\n")

    data = collect_public_data()

    print("Sources returned:")

    for source_name, rows in data.items():
        print(f"{source_name}: {len(rows)} rows")

    # Required source buckets
    required_sources = {
        "areas",
        "market_profile",
        "competition",
        "accessibility",
        "rent",
        "news_signals",
    }

    assert required_sources.issubset(data.keys()), (
        "Missing one or more required acquisition buckets."
    )

    # We have exactly three locked candidate areas.
    assert len(data["areas"]) == 3, (
        f"Expected 3 candidate areas, got {len(data['areas'])}"
    )

    # Confirm the three locked candidates exist.
    areas = {
        row["area"].strip().upper()
        for row in data["areas"]
        if row.get("area")
    }

    expected_areas = {"BKC", "DADAR", "VIKHROLI"}

    assert areas == expected_areas, (
        f"Expected {expected_areas}, got {areas}"
    )

    print("\nACQUISITION STRUCTURE PASSED")


if __name__ == "__main__":
    main()
