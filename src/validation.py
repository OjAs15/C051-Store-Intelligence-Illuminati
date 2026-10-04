def validate_payload(payload):
    valid=[]
    for row in payload.get("areas",[]):
        if all(k in row for k in ("area","latitude","longitude","catchment_radius_km")) and -90<=float(row["latitude"])<=90 and -180<=float(row["longitude"])<=180: valid.append(row)
    out=dict(payload); out["areas"]=valid; return out
