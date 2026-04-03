from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .modules.layers import (
    ForwardCacheRuntime,
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
from .modules.lora import LinearLora, replace_linear_with_lora


@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        collect: dict[object, object] | list[object] | set[object] | None = None,
        cache: dict[object, Tensor] | None = None,
        return_collected: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """
        Run one forward pass of Flux with optional tensor-level cache controls.

        Args:
            img: Image token sequence of shape (B, L_img, C_img).
            img_ids: Positional ids for image tokens.
            txt: Text token sequence of shape (B, L_txt, C_txt).
            txt_ids: Positional ids for text tokens.
            timesteps: Diffusion timesteps for the current step.
            y: Conditioning vector input.
            guidance: Guidance strength tensor when guidance embedding is enabled.
            collect: Optional cache-control config.
                - Mapping form (recommended): {cache_key: op}
                  where op is one of:
                    - "skip": return None for this tensor (fail-fast if later needed)
                    - "compute": compute normally, do not store output cache
                    - "compute_and_cache": compute and store in output cache
                    - "use_cache": read from input cache, error if missing
                - Iterable form: interpreted as keys with "compute_and_cache".
                Supported key formats:
                    - "stream:layer_idx:tensor_name"
                    - (stream, layer_idx, tensor_name)
            cache: Optional input cache mapping. Used by "use_cache" keys.
                Keys use the same format as collect. Values are tensors.
            return_collected: If True, return (output, generated_cache).
                generated_cache contains only tensors produced by
                "compute_and_cache" during this forward pass.

        Returns:
            - Tensor if return_collected is False.
            - (Tensor, Dict[str, Tensor]) if return_collected is True.

        Usage:
            1) Build input cache in one run:
               collect = {"double_stream:0:attn": "compute_and_cache"}
               out, new_cache = model(..., collect=collect, return_collected=True)

            2) Reuse cached tensor in a later run:
               collect = {"double_stream:0:attn": "use_cache"}
               out = model(..., collect=collect, cache=new_cache)
        """
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        cache_runtime = ForwardCacheRuntime(collect=collect, cache=cache)

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(
                img=img,
                txt=txt,
                vec=vec,
                pe=pe,
                cache_runtime=cache_runtime,
            )

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(
                img,
                vec=vec,
                pe=pe,
                cache_runtime=cache_runtime,
            )
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        if return_collected:
            return img, cache_runtime.collected_as_flat_dict()
        return img


class FluxLoraWrapper(Flux):
    def __init__(
        self,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lora_rank = lora_rank

        replace_linear_with_lora(
            self,
            max_rank=lora_rank,
            scale=lora_scale,
        )

    def set_lora_scale(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)
