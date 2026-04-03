from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, ClassVar

import torch
from einops import rearrange
from torch import Tensor

from flux.math import attention

CacheKey = tuple[str, int, str]  # (stream, layer, module)


@dataclass
class CacheConfig:
    enabled: bool = False
    update_interval: int = 1
    warmup_steps: int = 1
    method: str = "reuse"
    method_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CacheQuery:
    stream: str
    layer: int
    module: str
    step_index: int
    t_curr: float | None = None
    t_prev: float | None = None


_CACHE_REGISTRY: dict[str, type["FluxModuleCache"]] = {}


def register_cache_method(cache_cls: type["FluxModuleCache"]) -> type["FluxModuleCache"]:
    method_name = cache_cls.METHOD_NAME.strip().lower()
    if not method_name:
        raise ValueError("Cache method name cannot be empty.")
    _CACHE_REGISTRY[method_name] = cache_cls
    return cache_cls


def get_registered_cache_methods() -> tuple[str, ...]:
    return tuple(sorted(_CACHE_REGISTRY.keys()))


def resolve_cache_class(method: str) -> type["FluxModuleCache"]:
    method_name = method.strip().lower()
    if method_name not in _CACHE_REGISTRY:
        available = ", ".join(get_registered_cache_methods())
        raise ValueError(f"Unknown cache method '{method}'. Available methods: {available}")
    return _CACHE_REGISTRY[method_name]


class FluxModuleCache(ABC):
    """Base class for module-cache methods.

    Subclasses implement cache update/predict behavior and manage module-level caching policy.
    """

    METHOD_NAME: ClassVar[str] = "base"
    METHOD_PARAM_DEFAULTS: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        config: CacheConfig | None = None,
        update_selector: Callable[[CacheQuery], bool] | None = None,
    ) -> None:
        self.config = config or CacheConfig()
        self._update_selector = update_selector
        self._step_index = -1
        self._step_t_curr: float | None = None
        self._step_t_prev: float | None = None

        self.config.method = self.METHOD_NAME
        self.apply_method_params()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def reset(self) -> None:
        self._step_index = -1
        self._step_t_curr = None
        self._step_t_prev = None
        self.clear_state()

    @abstractmethod
    def clear_state(self) -> None:
        """Clear all cached states for a new denoising run."""

    def begin_step(self, step_index: int, t_curr: float | None = None, t_prev: float | None = None) -> None:
        self._step_index = int(step_index)
        self._step_t_curr = t_curr
        self._step_t_prev = t_prev

    @abstractmethod
    def run_module(
        self,
        stream: str,
        layer: int,
        module: str,
        **forward_kwargs: Any,
    ) -> Any:
        """Run or predict one module using full forward kwargs."""

    def make_query(self, stream: str, layer: int, module: str) -> CacheQuery:
        return CacheQuery(
            stream=stream,
            layer=int(layer),
            module=module,
            step_index=self._step_index,
            t_curr=self._step_t_curr,
            t_prev=self._step_t_prev,
        )

    def apply_method_params(self, method_params: dict[str, Any] | None = None) -> None:
        merged = dict(self.config.method_params)
        if method_params:
            merged.update(method_params)

        for key, value in self.METHOD_PARAM_DEFAULTS.items():
            merged.setdefault(key, value)

        self.validate_method_params(merged)
        self.config.method_params = merged

    @classmethod
    def validate_method_params(cls, params: dict[str, Any]) -> None:
        _ = params

    def get_method_param(self, name: str, default: Any = None) -> Any:
        return self.config.method_params.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self.config)
        out["method"] = self.METHOD_NAME
        return out

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "FluxModuleCache":
        method_value = config_dict.get("method")
        if not isinstance(method_value, str) or not method_value.strip():
            raise ValueError("Cache config must include a non-empty 'method' field.")

        method = method_value.strip().lower()
        cache_cls = resolve_cache_class(method)

        valid_fields = {field.name for field in fields(CacheConfig)}
        config_data = {key: value for key, value in config_dict.items() if key in valid_fields}
        config_data["method"] = cache_cls.METHOD_NAME

        return cache_cls(config=CacheConfig(**config_data))

    def save_config(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_config(cls, path: str | Path) -> "FluxModuleCache":
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Cache config must be a JSON object.")
        return cls.from_dict(loaded)

    def should_update(self, query: CacheQuery) -> bool:
        if self._step_index < 0:
            return True

        if not self.has_cache(query):
            return True

        if self._update_selector is not None:
            return bool(self._update_selector(query))

        if query.step_index < self.config.warmup_steps:
            return True

        interval = max(int(self.config.update_interval), 1)
        if interval <= 1:
            return True

        return query.step_index % interval == 0

    @abstractmethod
    def has_cache(self, query: CacheQuery) -> bool:
        """Return whether cache needed by predict already exists for the query."""

    @abstractmethod
    def update_cache(self, query: CacheQuery, value: Any) -> None:
        """Store fresh compute result into cache."""

    @abstractmethod
    def predict(self, query: CacheQuery) -> Any:
        """Predict value from cache when fresh compute is skipped."""

    @staticmethod
    def to_key(query: CacheQuery) -> CacheKey:
        return query.stream, query.layer, query.module

    def _detach_value(self, value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.detach()
        if isinstance(value, tuple):
            return tuple(self._detach_value(v) for v in value)
        if isinstance(value, list):
            return [self._detach_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._detach_value(v) for k, v in value.items()}
        return value

    def _clone_value(self, value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.clone()
        if isinstance(value, tuple):
            return tuple(self._clone_value(v) for v in value)
        if isinstance(value, list):
            return [self._clone_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._clone_value(v) for k, v in value.items()}
        return value

    def _sub(self, a: Any, b: Any) -> Any:
        if isinstance(a, Tensor) and isinstance(b, Tensor):
            return a - b
        if isinstance(a, tuple) and isinstance(b, tuple):
            return tuple(self._sub(x, y) for x, y in zip(a, b, strict=True))
        if isinstance(a, list) and isinstance(b, list):
            return [self._sub(x, y) for x, y in zip(a, b, strict=True)]
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                raise ValueError("Cannot subtract cache values with different dict keys.")
            return {k: self._sub(a[k], b[k]) for k in a}
        raise TypeError(f"Unsupported cache value type for subtraction: {type(a)}")

    def _add_scaled(self, a: Any, b: Any, scale: float) -> Any:
        if isinstance(a, Tensor) and isinstance(b, Tensor):
            return a + scale * b
        if isinstance(a, tuple) and isinstance(b, tuple):
            return tuple(self._add_scaled(x, y, scale) for x, y in zip(a, b, strict=True))
        if isinstance(a, list) and isinstance(b, list):
            return [self._add_scaled(x, y, scale) for x, y in zip(a, b, strict=True)]
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                raise ValueError("Cannot add cache values with different dict keys.")
            return {k: self._add_scaled(a[k], b[k], scale) for k in a}
        raise TypeError(f"Unsupported cache value type for scaled addition: {type(a)}")


def build_module_cache(
    enabled: bool = False,
    update_interval: int = 2,
    warmup_steps: int = 1,
    method: str | None = None,
    method_params: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> FluxModuleCache | None:
    if not enabled and config_path is None:
        return None

    if config_path is not None:
        cache = FluxModuleCache.load_config(config_path)
        if method is not None:
            desired_cls = resolve_cache_class(method)
            if not isinstance(cache, desired_cls):
                cache = desired_cls(config=cache.config)
    else:
        if method is None or not method.strip():
            raise ValueError("'method' is required when config_path is not provided.")

        resolved_method = method.strip().lower()
        cache_cls = resolve_cache_class(resolved_method)
        cache = cache_cls(config=CacheConfig(enabled=True, method=cache_cls.METHOD_NAME))

    cache.config.enabled = bool(enabled or cache.config.enabled)
    cache.config.update_interval = max(int(update_interval), 1)
    cache.config.warmup_steps = max(int(warmup_steps), 0)
    cache.config.method = cache.METHOD_NAME
    cache.apply_method_params(method_params)
    return cache


def parse_method_params_json(method_params_json: str | None) -> dict[str, Any] | None:
    if method_params_json is None:
        return None

    raw = method_params_json.strip()
    if raw == "":
        return None

    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("cache_method_params must be a JSON object string.")
    return loaded


@register_cache_method
class ReuseCache(FluxModuleCache):
    METHOD_NAME = "reuse"

    def __init__(
        self,
        config: CacheConfig | None = None,
        update_selector: Callable[[CacheQuery], bool] | None = None,
    ) -> None:
        super().__init__(config=config, update_selector=update_selector)
        self._entries: dict[CacheKey, Any] = {}

    def clear_state(self) -> None:
        self._entries.clear()

    def run_module(
        self,
        stream: str,
        layer: int,
        module: str,
        **forward_kwargs: Any,
    ) -> Any:
        if not self.enabled:
            return self._forward_module(stream=stream, module=module, **forward_kwargs)

        query = self.make_query(stream=stream, layer=layer, module=module)
        if self.should_update(query):
            value = self._forward_module(stream=stream, module=module, **forward_kwargs)
            self.update_cache(query, value)
            return value

        return self.predict(query)

    def _forward_module(self, stream: str, module: str, **forward_kwargs: Any) -> Any:
        if stream == "double_stream":
            block = forward_kwargs["block"]

            if module == "joint_attn":
                img = forward_kwargs["img"]
                txt = forward_kwargs["txt"]
                img_mod1 = forward_kwargs["img_mod1"]
                txt_mod1 = forward_kwargs["txt_mod1"]
                pe = forward_kwargs["pe"]

                img_modulated = block.img_norm1(img)
                img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
                img_qkv = block.img_attn.qkv(img_modulated)
                img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=block.num_heads)
                img_q, img_k = block.img_attn.norm(img_q, img_k, img_v)

                txt_modulated = block.txt_norm1(txt)
                txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
                txt_qkv = block.txt_attn.qkv(txt_modulated)
                txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=block.num_heads)
                txt_q, txt_k = block.txt_attn.norm(txt_q, txt_k, txt_v)

                q = torch.cat((txt_q, img_q), dim=2)
                k = torch.cat((txt_k, img_k), dim=2)
                v = torch.cat((txt_v, img_v), dim=2)

                attn = attention(q, k, v, pe=pe)
                txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]
                return block.img_attn.proj(img_attn), block.txt_attn.proj(txt_attn)

            if module == "img_mlp":
                img = forward_kwargs["img"]
                img_mod2 = forward_kwargs["img_mod2"]
                return block.img_mlp((1 + img_mod2.scale) * block.img_norm2(img) + img_mod2.shift)

            if module == "txt_mlp":
                txt = forward_kwargs["txt"]
                txt_mod2 = forward_kwargs["txt_mod2"]
                return block.txt_mlp((1 + txt_mod2.scale) * block.txt_norm2(txt) + txt_mod2.shift)

            raise ValueError(f"Unsupported module '{module}' for stream '{stream}'.")

        elif stream == "single_stream":
            if module != "total":
                raise ValueError(f"Unsupported module '{module}' for stream '{stream}'.")

            block = forward_kwargs["block"]
            x = forward_kwargs["x"]
            mod = forward_kwargs["mod"]
            pe = forward_kwargs["pe"]

            x_mod = (1 + mod.scale) * block.pre_norm(x) + mod.shift
            qkv, mlp = torch.split(block.linear1(x_mod), [3 * block.hidden_size, block.mlp_hidden_dim], dim=-1)

            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=block.num_heads)
            q, k = block.norm(q, k, v)

            attn = attention(q, k, v, pe=pe)
            return block.linear2(torch.cat((attn, block.mlp_act(mlp)), 2))

        raise ValueError(f"Unsupported stream '{stream}'.")

    def has_cache(self, query: CacheQuery) -> bool:
        return self.to_key(query) in self._entries

    def update_cache(self, query: CacheQuery, value: Any) -> None:
        self._entries[self.to_key(query)] = self._detach_value(value)

    def predict(self, query: CacheQuery) -> Any:
        key = self.to_key(query)
        if key not in self._entries:
            raise RuntimeError(f"No cache entry for {key}.")
        return self._clone_value(self._entries[key])


@register_cache_method
class LinearCache(FluxModuleCache):
    METHOD_NAME = "linear"
    METHOD_PARAM_DEFAULTS = {"linear_scale": 1.0}

    @classmethod
    def validate_method_params(cls, params: dict[str, Any]) -> None:
        if "linear_scale" not in params:
            return
        try:
            params["linear_scale"] = float(params["linear_scale"])
        except (TypeError, ValueError) as exc:
            raise ValueError("linear_scale must be a float-convertible value.") from exc

    def __init__(
        self,
        config: CacheConfig | None = None,
        update_selector: Callable[[CacheQuery], bool] | None = None,
    ) -> None:
        super().__init__(config=config, update_selector=update_selector)
        self._history: dict[CacheKey, list[Any]] = {}

    def clear_state(self) -> None:
        self._history.clear()

    def run_module(
        self,
        stream: str,
        layer: int,
        module: str,
        **forward_kwargs: Any,
    ) -> Any:
        if not self.enabled:
            return self._forward_module(stream=stream, module=module, **forward_kwargs)

        query = self.make_query(stream=stream, layer=layer, module=module)
        if self.should_update(query):
            value = self._forward_module(stream=stream, module=module, **forward_kwargs)
            self.update_cache(query, value)
            return value

        return self.predict(query)

    def _forward_module(self, stream: str, module: str, **forward_kwargs: Any) -> Any:
        if stream == "double_stream":
            block = forward_kwargs["block"]

            if module == "joint_attn":
                img = forward_kwargs["img"]
                txt = forward_kwargs["txt"]
                img_mod1 = forward_kwargs["img_mod1"]
                txt_mod1 = forward_kwargs["txt_mod1"]
                pe = forward_kwargs["pe"]

                img_modulated = block.img_norm1(img)
                img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
                img_qkv = block.img_attn.qkv(img_modulated)
                img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=block.num_heads)
                img_q, img_k = block.img_attn.norm(img_q, img_k, img_v)

                txt_modulated = block.txt_norm1(txt)
                txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
                txt_qkv = block.txt_attn.qkv(txt_modulated)
                txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=block.num_heads)
                txt_q, txt_k = block.txt_attn.norm(txt_q, txt_k, txt_v)

                q = torch.cat((txt_q, img_q), dim=2)
                k = torch.cat((txt_k, img_k), dim=2)
                v = torch.cat((txt_v, img_v), dim=2)

                attn = attention(q, k, v, pe=pe)
                txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]
                return block.img_attn.proj(img_attn), block.txt_attn.proj(txt_attn)

            if module == "img_mlp":
                img = forward_kwargs["img"]
                img_mod2 = forward_kwargs["img_mod2"]
                return block.img_mlp((1 + img_mod2.scale) * block.img_norm2(img) + img_mod2.shift)

            if module == "txt_mlp":
                txt = forward_kwargs["txt"]
                txt_mod2 = forward_kwargs["txt_mod2"]
                return block.txt_mlp((1 + txt_mod2.scale) * block.txt_norm2(txt) + txt_mod2.shift)

            raise ValueError(f"Unsupported module '{module}' for stream '{stream}'.")

        if stream == "single_stream":
            if module != "total":
                raise ValueError(f"Unsupported module '{module}' for stream '{stream}'.")

            block = forward_kwargs["block"]
            x = forward_kwargs["x"]
            mod = forward_kwargs["mod"]
            pe = forward_kwargs["pe"]

            x_mod = (1 + mod.scale) * block.pre_norm(x) + mod.shift
            qkv, mlp = torch.split(block.linear1(x_mod), [3 * block.hidden_size, block.mlp_hidden_dim], dim=-1)

            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=block.num_heads)
            q, k = block.norm(q, k, v)

            attn = attention(q, k, v, pe=pe)
            return block.linear2(torch.cat((attn, block.mlp_act(mlp)), 2))

        raise ValueError(f"Unsupported stream '{stream}'.")

    def has_cache(self, query: CacheQuery) -> bool:
        history = self._history.get(self.to_key(query), [])
        return len(history) > 0

    def update_cache(self, query: CacheQuery, value: Any) -> None:
        key = self.to_key(query)
        history = self._history.setdefault(key, [])
        history.append(self._detach_value(value))
        if len(history) > 2:
            del history[:-2]

    def predict(self, query: CacheQuery) -> Any:
        key = self.to_key(query)
        history = self._history.get(key)
        if not history:
            raise RuntimeError(f"No cache entry for {key}.")

        last = history[-1]
        if len(history) < 2:
            return self._clone_value(last)

        prev = history[-2]
        scale = float(self.get_method_param("linear_scale", 1.0))
        return self._add_scaled(last, self._sub(last, prev), scale)
