import os
from datetime import datetime

NOW = None
if os.environ.get("CHEMELEON_SHARED_OUTPUT_DIR", False):
    NOW = os.environ["CHEMELEON_SHARED_OUTPUT_DIR"]
else:
    NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.environ["CHEMELEON_SHARED_OUTPUT_DIR"] = NOW
