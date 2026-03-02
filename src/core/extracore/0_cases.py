
from .config import CaseConfig

def make_default_cases():
    return {"const": CaseConfig(name="const", c0=1.0)}
