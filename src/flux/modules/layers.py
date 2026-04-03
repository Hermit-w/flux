import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

import torch
from einops import rearrange
from torch import Tensor, nn

from flux.math import attention, rope


CollectOp = Literal["skip", "compute", "compute_and_cache"]
Stage = Literal["full", "cache"]
VALID_COLLECT_OPS = {"skip", "compute", "compute_and_cache"}


@dataclass(frozen=True)
class CacheKey:
    stream: str
    layer_idx: int
    tensor_name: str

    @classmethod
    def parse(cls, key: object) -> "CacheKey":
        if isinstance(key, cls):
            return key
        if isinstance(key, tuple) and len(key) == 3:
            stream, layer_idx, tensor_name = key
            return cls(stream=str(stream), layer_idx=int(layer_idx), tensor_name=str(tensor_name))
        if isinstance(key, str):
            parts = key.split(":", 2)
            if len(parts) != 3:
                raise ValueError(
                    "Cache key string must use format 'stream:layer_idx:tensor_name', "
                    f"but got: {key}"
                )
            stream, layer_idx, tensor_name = parts
            return cls(stream=stream, layer_idx=int(layer_idx), tensor_name=tensor_name)
        raise TypeError(
            "Cache key must be CacheKey, tuple[str, int, str], or string "
            "'stream:layer_idx:tensor_name'."
        )

    def flat(self) -> str:
        return f"{self.stream}:{self.layer_idx}:{self.tensor_name}"


STREAM_TENSOR_EXECUTION_ORDER: dict[str, tuple[str, ...]] = {
    "double_stream": (
        "img_modulated",
        "img_qkv",
        "img_q_norm",
        "img_k_norm",
        "txt_modulated",
        "txt_qkv",
        "txt_q_norm",
        "txt_k_norm",
        "q",
        "k",
        "v",
        "attn",
        "img_attn_proj",
        "img_after_attn",
        "img_mlp_in",
        "img_mlp_out",
        "img_out",
        "txt_attn_proj",
        "txt_after_attn",
        "txt_mlp_in",
        "txt_mlp_out",
        "txt_out",
    ),
    "single_stream": (
        "x_mod",
        "linear1_out",
        "q_norm",
        "k_norm",
        "attn",
        "mlp_act",
        "output",
        "x_out",
    ),
}


# Full set of collectable tensors (per stream), derived from execution order.
ALL_COLLECTABLE_TENSORS: dict[str, tuple[str, ...]] = STREAM_TENSOR_EXECUTION_ORDER


def _parse_collect_op(raw_op: object) -> CollectOp:
    if raw_op not in VALID_COLLECT_OPS:
        raise ValueError(f"Unsupported collect op: {raw_op}")
    return cast(CollectOp, raw_op)


def build_collect_keys(
    stream: str,
    layer_idx: int,
    tensor_names: Iterable[str] | None = None,
    flat: bool = True,
) -> list[str] | list[tuple[str, int, str]]:
    names = tuple(tensor_names) if tensor_names is not None else ALL_COLLECTABLE_TENSORS.get(stream, ())
    if flat:
        return [f"{stream}:{layer_idx}:{name}" for name in names]
    return [(stream, layer_idx, name) for name in names]


def build_collect_operations(
    stream: str,
    layer_idx: int,
    tensor_names: Iterable[str] | None = None,
    op: CollectOp = "compute_and_cache",
    flat: bool = True,
) -> dict[object, CollectOp]:
    keys = build_collect_keys(stream=stream, layer_idx=layer_idx, tensor_names=tensor_names, flat=flat)
    return {key: op for key in keys}


class ForwardCacheRuntime:
    def __init__(
        self,
        collect: Mapping[object, object] | Iterable[object] | None = None,
        cache: Mapping[object, Tensor] | None = None,
        stage: Stage = "full",
    ):
        self.cache_storage: dict[CacheKey, Tensor] = {}
        self.operations: dict[CacheKey, CollectOp] = {}
        self.generated_cache: dict[CacheKey, Tensor] = {}
        self.stage = stage

        # if isinstance(collect, Mapping):
        #     validate_collect_config(collect=collect, cache=cache)

        if cache is not None:
            for key, value in cache.items():
                self.cache_storage[CacheKey.parse(key)] = value

        if collect is None:
            pass
        elif isinstance(collect, Mapping):
            for key, op in collect.items():
                self.operations[CacheKey.parse(key)] = _parse_collect_op(op)
        elif isinstance(collect, Iterable) and not isinstance(collect, (str, bytes)):
            for key in collect:
                self.operations[CacheKey.parse(key)] = "compute_and_cache"
        else:
            raise TypeError("collect must be None, a mapping of key->op, or an iterable of keys.")

    def resolve(
        self,
        stream: str,
        layer_idx: int,
        tensor_name: str,
        compute_fn: Callable[[], Tensor],
    ) -> Tensor:
        key = CacheKey(stream=stream, layer_idx=layer_idx, tensor_name=tensor_name)
        op = self.operations.get(key, "compute")
        
        if self.stage == "full":
            value = compute_fn()
            if op == "compute_and_cache":
                self.generated_cache[key] = value
            return value
        assert self.stage == "cache", f"Unsupported stage: {self.stage}"

        if op == "skip":
            # Intentional hard-skip: return None placeholder and let downstream fail fast
            # if this tensor is actually required by subsequent computation.
            return cast(Tensor, None)
        elif op == "compute_and_cache":
            if key in self.cache_storage:
                return self.cache_storage[key]
            else:
                raise KeyError(f"Cache key not found for compute_and_cache op: {key.flat()} at stage 'cache'")
        elif op == "compute":
            value = compute_fn()
            return value
        else:
            raise ValueError(f"Unsupported collect op: {op}")

    def collected_as_flat_dict(self) -> dict[str, Tensor]:
        return {key.flat(): value for key, value in self.generated_cache.items()}


def resolve_cached_tensor(
    cache_runtime: ForwardCacheRuntime | None,
    stream: str,
    layer_idx: int,
    tensor_name: str,
    compute_fn: Callable[[], Tensor],
) -> Tensor:
    if cache_runtime is None:
        return compute_fn()
    return cache_runtime.resolve(stream=stream, layer_idx=layer_idx, tensor_name=tensor_name, compute_fn=compute_fn)


def resolve_cached_tensors(
    cache_runtime: ForwardCacheRuntime | None,
    stream: str,
    layer_idx: int,
    compute_fns: Mapping[str, Callable[[], Tensor]],
) -> dict[str, Tensor]:
    return {
        name: resolve_cached_tensor(
            cache_runtime=cache_runtime,
            stream=stream,
            layer_idx=layer_idx,
            tensor_name=name,
            compute_fn=compute_fn,
        )
        for name, compute_fn in compute_fns.items()
    }


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)

    def normalize_query(self, q: Tensor, v: Tensor) -> Tensor:
        return self.query_norm(q).to(v)

    def normalize_key(self, k: Tensor, v: Tensor) -> Tensor:
        return self.key_norm(k).to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        layer_idx: int = -1,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        cache_runtime: ForwardCacheRuntime | None = None,
    ) -> tuple[Tensor, Tensor]:
        stream = "double_stream"

        def cached(name: str, compute_fn: Callable[[], Tensor]) -> Tensor:
            return resolve_cached_tensor(
                cache_runtime=cache_runtime,
                stream=stream,
                layer_idx=self.layer_idx,
                tensor_name=name,
                compute_fn=compute_fn,
            )

        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = cached("img_modulated", lambda: (1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift)
        img_qkv = cached("img_qkv", lambda: self.img_attn.qkv(img_modulated))
        if img_qkv is None:
            img_q_raw, img_k_raw, img_v = cast(Tensor, None), cast(Tensor, None), cast(Tensor, None)
        else:
            img_q_raw, img_k_raw, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_normed = resolve_cached_tensors(
            cache_runtime=cache_runtime,
            stream=stream,
            layer_idx=self.layer_idx,
            compute_fns={
                "img_q_norm": lambda: self.img_attn.norm.normalize_query(img_q_raw, img_v),
                "img_k_norm": lambda: self.img_attn.norm.normalize_key(img_k_raw, img_v),
            },
        )
        if img_normed is None:
            img_q, img_k = cast(Tensor, None), cast(Tensor, None)
        else:
            img_q = img_normed["img_q_norm"]
            img_k = img_normed["img_k_norm"]

        # prepare txt for attention
        txt_modulated = cached("txt_modulated", lambda: (1 + txt_mod1.scale) * self.txt_norm1(txt) + txt_mod1.shift)
        txt_qkv = cached("txt_qkv", lambda: self.txt_attn.qkv(txt_modulated))
        if txt_qkv is None:
            txt_q_raw, txt_k_raw, txt_v = cast(Tensor, None), cast(Tensor, None), cast(Tensor, None)
        else:
            txt_q_raw, txt_k_raw, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_normed = resolve_cached_tensors(
            cache_runtime=cache_runtime,
            stream=stream,
            layer_idx=self.layer_idx,
            compute_fns={
                "txt_q_norm": lambda: self.txt_attn.norm.normalize_query(txt_q_raw, txt_v),
                "txt_k_norm": lambda: self.txt_attn.norm.normalize_key(txt_k_raw, txt_v),
            },
        )
        if txt_normed is None:
            txt_q, txt_k = cast(Tensor, None), cast(Tensor, None)
        else:
            txt_q = txt_normed["txt_q_norm"]
            txt_k = txt_normed["txt_k_norm"]

        # run actual attention
        q = cached("q", lambda: torch.cat((txt_q, img_q), dim=2))
        k = cached("k", lambda: torch.cat((txt_k, img_k), dim=2))
        v = cached("v", lambda: torch.cat((txt_v, img_v), dim=2))

        attn = cached("attn", lambda: attention(q, k, v, pe=pe))
        if attn is None:
            txt_attn, img_attn = cast(Tensor, None), cast(Tensor, None)
        else:
            txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img blocks
        img_attn_proj = cached("img_attn_proj", lambda: self.img_attn.proj(img_attn))
        img_after_attn = cached("img_after_attn", lambda: img + img_mod1.gate * img_attn_proj)
        img_mlp_in = cached(
            "img_mlp_in",
            lambda: (1 + img_mod2.scale) * self.img_norm2(img_after_attn) + img_mod2.shift,
        )
        img_mlp_out = cached("img_mlp_out", lambda: self.img_mlp(img_mlp_in))
        img = cached("img_out", lambda: img_after_attn + img_mod2.gate * img_mlp_out)

        # calculate the txt blocks
        txt_attn_proj = cached("txt_attn_proj", lambda: self.txt_attn.proj(txt_attn))
        txt_after_attn = cached("txt_after_attn", lambda: txt + txt_mod1.gate * txt_attn_proj)
        txt_mlp_in = cached(
            "txt_mlp_in",
            lambda: (1 + txt_mod2.scale) * self.txt_norm2(txt_after_attn) + txt_mod2.shift,
        )
        txt_mlp_out = cached("txt_mlp_out", lambda: self.txt_mlp(txt_mlp_in))
        txt = cached("txt_out", lambda: txt_after_attn + txt_mod2.gate * txt_mlp_out)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        layer_idx: int = -1,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        cache_runtime: ForwardCacheRuntime | None = None,
    ) -> Tensor:
        stream = "single_stream"

        def cached(name: str, compute_fn: Callable[[], Tensor]) -> Tensor:
            return resolve_cached_tensor(
                cache_runtime=cache_runtime,
                stream=stream,
                layer_idx=self.layer_idx,
                tensor_name=name,
                compute_fn=compute_fn,
            )

        mod, _ = self.modulation(vec)
        x_mod = cached("x_mod", lambda: (1 + mod.scale) * self.pre_norm(x) + mod.shift)
        linear1_out = cached("linear1_out", lambda: self.linear1(x_mod))
        if linear1_out is None:
            qkv, mlp = cast(Tensor, None), cast(Tensor, None)
        else:
            qkv, mlp = torch.split(linear1_out, [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        if qkv is None:
            q_raw, k_raw, v = cast(Tensor, None), cast(Tensor, None), cast(Tensor, None)
        else:
            q_raw, k_raw, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        qk_normed = resolve_cached_tensors(
            cache_runtime=cache_runtime,
            stream=stream,
            layer_idx=self.layer_idx,
            compute_fns={
                "q_norm": lambda: self.norm.normalize_query(q_raw, v),
                "k_norm": lambda: self.norm.normalize_key(k_raw, v),
            },
        )
        if qk_normed is None:
            q, k = cast(Tensor, None), cast(Tensor, None)
        else:
            q = qk_normed["q_norm"]
            k = qk_normed["k_norm"]

        # compute attention
        attn = cached("attn", lambda: attention(q, k, v, pe=pe))
        # compute activation in mlp stream, cat again and run second linear layer
        mlp_act = cached("mlp_act", lambda: self.mlp_act(mlp))
        output = cached("output", lambda: self.linear2(torch.cat((attn, mlp_act), 2)))
        x = cached("x_out", lambda: x + mod.gate * output)
        return x


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
