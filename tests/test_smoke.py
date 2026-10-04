from src.config import CANDIDATES, EXISTING_STORES
from src.pipeline import refresh_v2_pipeline
def test_locked_setup(): assert len(CANDIDATES)==3 and len(EXISTING_STORES)==2
def test_pipeline():
 r=refresh_v2_pipeline(); assert len(r["ranking"])==3
