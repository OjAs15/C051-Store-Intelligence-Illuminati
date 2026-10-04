import math
from src.model import train_demo_demand_model, predict_demand

def _clip(v): return max(0,min(1,v))
def _hav(lat1,lon1,lat2,lon2):
    r=6371; p1=math.radians(lat1); p2=math.radians(lat2); dp=math.radians(lat2-lat1); dl=math.radians(lon2-lon1); a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2; return 2*r*math.asin(math.sqrt(a))
def run_all_engines(row, stores):
    mpi=50 if any(row.get(k) is None for k in ("population","purchasing_power_proxy","activity_proxy")) else sum(float(row[k]) for k in ("population","purchasing_power_proxy","activity_proxy"))/3
    pred=max(0,6000+predict_demand(train_demo_demand_model(),row)*1800)
    comp=row.get("competition_count"); acc=row.get("accessibility_index"); cf=.8 if comp is None else _clip(1-float(comp)/10); af=.85 if acc is None else _clip(float(acc)); captured=pred*cf*af
    cann=0
    for s in stores:
        cann += 900*math.exp(-.55*_hav(row["latitude"],row["longitude"],s["lat"],s["lon"]))
    inc=max(0,captured-cann); rent=row.get("rent_benchmark"); sr=None if rent in (None,0) else captured/float(rent)
    return {"market_potential":round(mpi,1),"predicted_demand":round(pred),"captured_demand":round(captured),"cannibalisation":round(cann),"incremental_demand":round(inc),"sales_rent":"N/A" if sr is None else round(sr,2)}
