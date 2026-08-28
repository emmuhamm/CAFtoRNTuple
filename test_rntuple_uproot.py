import uproot
import numpy as np
import awkward as ak
import sys
import time

t = time.time()

f = uproot.open(sys.argv[1])
caf = f["caf"].arrays()
part = caf.rec.common.ixn.dlp.part.dlp
muons = part[(part.pdg == 13) | (part.pdg == -13)]
np.histogram(ak.flatten(muons.E, axis=None), bins=50)

print(time.time() - t)

# ------------- #

t = time.time()

part = f["caf"]["rec.common.ixn.dlp.part.dlp"].arrays().dlp
muons = part[(part.pdg == 13) | (part.pdg == -13)]
np.histogram(ak.flatten(muons.E, axis=None), bins=50)

print(time.time() - t)