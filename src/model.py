import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

def train_demo_demand_model():
    rng=np.random.default_rng(42); X=rng.normal(size=(300,6)); y=.30*X[:,0]+.25*X[:,1]+.25*X[:,2]-.15*X[:,3]+.12*X[:,4]+.08*X[:,5]+rng.normal(scale=.15,size=300)
    m=HistGradientBoostingRegressor(random_state=42).fit(X,y); return {"model":m,"training_type":"synthetic_demo"}

def predict_demand(bundle,row):
    safe=lambda v: 0.0 if v is None else float(v)
    X=np.array([[safe(row.get("population")),safe(row.get("purchasing_power_proxy")),safe(row.get("activity_proxy")),safe(row.get("competition_count")),safe(row.get("accessibility_index")),safe(row.get("commercial_intensity"))]])
    return float(bundle["model"].predict(X)[0])
