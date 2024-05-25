import sys
import os
import inspect
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(Path(__file__).parent.parent), ""))
from squice import DataLoaders as cc

from os.path import dirname, abspath
import sys

DIR = dirname(abspath(__file__))
print(DIR)


# ---------------------------------------------------------------------------
def test_numpy():
    print(f"Testing utility: {inspect.stack()[0][3]}")
    npf = cc.NumpyFile(f"{DIR}/data/data1.npy")
    npf.load()
    print(npf.mtx)
    assert npf.mtx[0][0] == 1, npf.mtx[0][0]
    assert npf.mtx[0][2] == 3, npf.mtx[0][2]
    assert npf.mtx[2][0] == 7, npf.mtx[2][0]
    assert npf.mtx[1][1] == 5, npf.mtx[1][1]
    assert npf.mtx[2][2] == 9, npf.mtx[2][2]


###########################################################################
if __name__ == "__main__":
    test_numpy()
    # test_squared()
