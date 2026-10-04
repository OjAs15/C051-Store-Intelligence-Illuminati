from src.engines import run_all_engines
def rank_candidates(features, stores):
    rows=[]
    for k,row in features.items(): rows.append({"candidate_id":k,"area":row["area"],**run_all_engines(row,stores)})
    return sorted(rows,key=lambda r:r["incremental_demand"],reverse=True)
