from collections import defaultdict, deque
import numpy as np
import torch
import torch.nn as nn

from typing import Iterable, Any, Optional


def _first_tensor(x: Any) -> Optional[torch.Tensor]:
    """Recursively find the first tensor in a nested structure."""
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        for v in x:
            tensor = _first_tensor(v)
            if tensor is not None:
                return tensor
    if isinstance(x, dict):
        for v in x.values():
            tensor = _first_tensor(v)
            if tensor is not None:
                return tensor
        return


class ActivationHook:
    def __init__(
        self,
        model: nn.Module,
        layers_to_hook: Iterable[str],
        hist_max_samples: int = 4096,
        sample_per_call: int = 512,
        every_n: int = 1,
    ):
        self.model = model
        self.layers_to_hook = list(layers_to_hook)
        self.hist_max_samples = hist_max_samples
        self.sample_per_call = int(sample_per_call)
        self.every_n = max(1, int(every_n))

        self.stats: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.counts: dict[str, int] = defaultdict(int)

        self.samples: dict[str, deque] = {
            name: deque(maxlen=self.hist_max_samples) for name in self.layers_to_hook
        }

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._batch_counter: int = 0

    @staticmethod
    def _stats(t: torch.Tensor) -> dict[str, float]:
        x = t.detach()
        if not torch.is_tensor(x) or x.numel() == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "sparsity": 0.0}
        if x.is_floating_point() and x.dtype != torch.float32:
            x = x.float()
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        return {
            "mean": x.mean().item(),
            "std": x.std(unbiased=False).item(),
            "min": x.amin().item(),
            "max": x.amax().item(),
            "sparsity": (x == 0).float().mean().item(),
        }

    def _hook_fn(self, name, module: nn.Module, inp: Any, out: Any):
        if self.layers_to_hook and name == self.layers_to_hook[0]:
            self._batch_counter += 1

        if (self._batch_counter % self.every_n) != 0:
            return

        act = _first_tensor(out)
        if act is None:
            return

        stats = self._stats(act)
        for stat_name, value in stats.items():
            self.stats[name][stat_name] += value
        self.counts[name] += 1

        act_vec = act.detach()
        if act_vec.is_floating_point() and act_vec.dtype != torch.float32:
            act_vec = act_vec.float()
        act_vec = act.reshape(-1)
        if act_vec.numel():
            take = min(self.sample_per_call, act_vec.numel())
            idx = torch.randint(0, act_vec.numel(), (take,), device=act_vec.device)
            self.samples[name].extend(act_vec[idx].cpu().numpy().tolist())

    def summarize_means(self, prefix="debug/act") -> dict[str, float]:
        """Averages the collected stats and resets them for the next use."""
        averaged_stats: dict[str, float] = {}
        for layer_name, data in self.stats.items():
            count = max(self.counts[layer_name], 1)
            for stat_name, value in data.items():
                averaged_stats[f"{prefix}/{layer_name}/{stat_name}"] = value / count

        self.stats.clear()
        self.counts.clear()
        return averaged_stats

    def summarize_hists(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, buf in self.samples.items():
            if len(buf) > 0:
                out[name] = np.asarray(buf, dtype=np.float32)
            buf.clear()
        return out

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        name_to_module = dict(self.model.named_modules())
        for name in self.layers_to_hook:
            module = name_to_module.get(name)
            if module is not None:
                handle = module.register_forward_hook(
                    lambda m, i, o, layer_name=name: self._hook_fn(layer_name, m, i, o)
                )
                self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
