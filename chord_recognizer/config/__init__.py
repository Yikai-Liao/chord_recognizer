import numpy as np
from .gen_config import gen_chord_config
import os

_CUR_DIR = os.path.split(__file__)[0]
_CHORD_CONFIG_PATH = os.path.join(_CUR_DIR, "chord_config.npz")
_GEN_CONFIG_CODE_PATH = os.path.join(_CUR_DIR, "gen_config.py")
with open(_GEN_CONFIG_CODE_PATH, encoding="UTF-8") as f:
    _CODE_HASH = hash(f.read())

if not (os.path.isfile(_CHORD_CONFIG_PATH) and np.load(_CHORD_CONFIG_PATH)['hash'] == _CODE_HASH):
    CHORD_CONFIG = gen_chord_config()
    np.savez(_CHORD_CONFIG_PATH, hash=_CODE_HASH, **CHORD_CONFIG)
else:
    CHORD_CONFIG = {**np.load(_CHORD_CONFIG_PATH)}

CHORD_CONFIG['pitch'] = [chroma.nonzero()[0].tolist() for chroma in CHORD_CONFIG['chroma']]
__all__ = ["CHORD_CONFIG"]
