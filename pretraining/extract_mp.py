# extract_mp.py
import sys
from pathlib import Path

import torch

from train import MPNN  # import from here to trigger other imports
from config import MP_ACTIVATION


try:
    ckpt = Path(sys.argv[1])
except:
    print("usage: python extract_mp.py /path/to/model.ckpt")
    exit(1)

out = ckpt.parent.resolve() / (ckpt.stem + "_mp.pt")
print(f"output will be written to '{out}'")
if out.exists():
    print("output file already exists - exiting.")
    exit(1)

m = MPNN.load_from_checkpoint(ckpt, map_location="cpu")
hps = dict(m.message_passing.hparams)
hps.pop("cls")
hps["activation"] = MP_ACTIVATION
torch.save({"hyper_params": hps, "state_dict": m.message_passing.state_dict()}, out)
