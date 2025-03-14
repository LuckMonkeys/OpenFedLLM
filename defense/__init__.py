from .krum import Krum
from .foolsgold import FoolsGold
from .rflbat import RFLBAT
from .crfl import CRFL
from .dp import DP
from .median import Median
from .nc import NormClipping
from .sfed import SparseFed
from .trimmed_mean import TrimmedMean
from .fedavg import FedAvg

from dataclasses import asdict, is_dataclass
from omegaconf import DictConfig
from .base import vectorize_dict

DEFENDERS = {
    "fedavg": FedAvg,
    "krum": Krum,
    "multi-krum": Krum,
    "foolsgold": FoolsGold,
    "rflbat": RFLBAT,
    "crfl": CRFL,
    "dp": DP,
    "median": Median,
    "nc": NormClipping,
    "sfed": SparseFed,
    "trimmed_mean": TrimmedMean
}

def load_defender(args):
    if is_dataclass(args):
        return DEFENDERS[args.name.lower()](**asdict(args))
    elif isinstance(args, dict) or isinstance(args, DictConfig):
        return DEFENDERS[args["name"].lower()](**args)
    else:
        raise ValueError("Invalid type of args")
