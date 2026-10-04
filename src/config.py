from pathlib import Path
APP_TITLE="Store Expansion Intelligence"
BASE_DIR=Path(__file__).resolve().parents[1]
DATA_DIR=BASE_DIR/"data"/"v2"
RAW_DIR=DATA_DIR/"raw"
PROCESSED_DIR=DATA_DIR/"processed"
OUTPUT_DIR=DATA_DIR/"output"
CATCHMENT_RADIUS_KM=2.0
EXISTING_STORES=[
 {"id":"PRABHADEVI","name":"Croma Prabhadevi","lat":19.01289,"lon":72.82364},
 {"id":"POWAI","name":"Croma Powai","lat":19.08449,"lon":72.88434},
]
CANDIDATES=[
 {"id":"BKC","name":"BKC","lat":19.0671,"lon":72.8657},
 {"id":"DADAR","name":"Dadar","lat":19.0239,"lon":72.8382},
 {"id":"VIKHROLI","name":"Vikhroli","lat":19.1115,"lon":72.9280},
]
