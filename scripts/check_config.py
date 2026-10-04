from src.config import (
    EXISTING_STORES,
    CANDIDATES,
    CATCHMENT_RADIUS_KM,
)

print("Catchment:", CATCHMENT_RADIUS_KM, "km")

print("\nExisting stores:")
for store in EXISTING_STORES:
    print(
        store["id"],
        "|",
        store["name"],
        "|",
        store["lat"],
        store["lon"],
    )

print("\nCandidates:")
for candidate in CANDIDATES:
    print(
        candidate["id"],
        "|",
        candidate["name"],
        "|",
        candidate["lat"],
        candidate["lon"],
    )

assert len(EXISTING_STORES) == 2
assert len(CANDIDATES) == 3
assert CATCHMENT_RADIUS_KM == 2.0

print("\nCONFIG CHECK PASSED")
