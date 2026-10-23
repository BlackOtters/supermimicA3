import torch
import math
import numbers
from functools import wraps
from typing import Sequence, Dict, Any
from tensordict import TensorDictBase, TensorDict
from prettytable import PrettyTable


def drop_nonfinite_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return logging metrics with NaN/Inf numeric values omitted.

    This is a logging boundary, not a replacement for environment/PPO
    integrity checks.  Its job is to prevent one unexpected plugin metric
    from writing a permanent NaN point to a W&B curve after the training
    state itself has already been isolated.  The replacement count remains
    visible under ``env/nonfinite_metric_values_dropped``.
    """
    result: Dict[str, Any] = {}
    dropped = 0
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
                dropped += 1
                continue
        elif isinstance(value, numbers.Number) and not isinstance(value, bool):
            if not math.isfinite(value):
                dropped += 1
                continue
        result[key] = value
    if dropped:
        result["env/nonfinite_metric_values_dropped"] = float(dropped)
    return result


def table_print(info: Dict[str, Any]):
    pt = PrettyTable()
    nrow = max(len(v) for v in info.values())
    for k, v in info.items():
        data = [f"{kk}:{vv:.3f}" for kk, vv in v.items()]
        data += [" "] * (nrow - len(data))
        pt.add_column(k, data)
    print(pt)


def batchify(func, broadcast=True):
    @wraps(func)
    def wrapped(*args, **kwargs):
        batch_shapes = [arg.shape[:-1] for arg in args]
        if broadcast:
            batch_shape = torch.broadcast_shapes(*batch_shapes)
        else:
            batch_shape = set(batch_shapes)
            if len(batch_shape) != 1:
                raise ValueError()
            batch_shape = batch_shape.pop()
        args = [
            arg.expand(*batch_shape, arg.shape[-1]).reshape(-1, arg.shape[-1]) 
            for arg in args
        ]
        ret = func(*args, **kwargs)
        return ret.reshape(*batch_shape, *ret.shape[1:])
    return wrapped

class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1

class EpisodeStats:
    def __init__(self, in_keys: Sequence[str], device: torch.device):
        self.in_keys = in_keys
        self.device = device
        self._stats = TensorDict({key: torch.tensor([0.], device=device) for key in in_keys}, [1])
        self._episodes = torch.tensor(0, device=device)

    def add(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_tensordict = tensordict["next"]
        done = next_tensordict["done"]
        if done.any():
            done = done.squeeze(-1)
            next_tensordict = next_tensordict.select(*self.in_keys)
            self._stats = self._stats + next_tensordict[done].sum(dim=0)
            self._episodes += done.sum()
        return len(self)
    
    def pop(self):
        stats = self._stats / self._episodes
        self._stats.zero_()
        self._episodes.zero_()
        return stats.cpu()

    def __len__(self):
        return self._episodes.item()
