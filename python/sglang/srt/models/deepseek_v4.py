from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.jit_kernel.deepseek_v4 import (
    fused_norm_rope_inplace,
    fused_q_norm_rope,
    fused_rope_inplace,
)
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attention.dsv4.compressor import Compressor
from sglang.srt.layers.attention.dsv4.indexer import C4Indexer
from sglang.srt.layers.attention.nsa.utils import (
    can_nsa_cp_split,
    is_nsa_enable_prefill_cp,
    is_nsa_prefill_cp_round_robin_split,
    nsa_use_prefill_cp,
)
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dp_attention import (
    _DpGatheredBufferWrapper,
    attn_tp_all_gather,
    dp_gather_partial,
    dp_scatter,
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    prepare_context_parallel_metadata,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.mem_cache.memory_pool import RadixAttention
from sglang.srt.model_executor.cuda_graph_runner import (
    compile_in_capture_mode,
    get_is_capture_mode,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_loader.utils import maybe_executor_submit, should_async_load
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dbrx import ReplicatedLinear
from sglang.srt.models.deepseek_v2 import ParallelLMHead, _is_cuda, _is_hip, _is_npu
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    LazyValue,
    add_prefix,
    log_info_on_rank0,
    make_layers,
)
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)

_FP8_WO_A_GEMM = envs.SGLANG_OPT_FP8_WO_A_GEMM.get()


# todo https://mp.weixin.qq.com/s/ZsDoLeSvYMK15-OtVyI66A

if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        DeepseekV4AttnBackend,
    )
    from sglang.srt.layers.quantization import QuantizationConfig
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@triton.jit
def _rms_normalize_kernel(
    x_ptr,
    weight_ptr,
    eps,
    stride_row,
    dim,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < dim

    base = pid * stride_row
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / dim
    rms_inv = tl.rsqrt(mean_sq + eps)
    out = x * rms_inv

    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        out = out * weight

    tl.store(x_ptr + base + offs, out, mask=mask)


def rms_normalize_triton(
    x: torch.Tensor, eps: float, weight: torch.Tensor = None
) -> torch.Tensor:
    dim = x.shape[-1]
    x_flat = x.view(-1, dim)
    num_rows = x_flat.shape[0]

    BLOCK_SIZE = triton.next_power_of_2(dim)
    grid = (num_rows,)

    _rms_normalize_kernel[grid](
        x_flat,
        weight,
        eps,
        x_flat.stride(0),
        dim,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=(weight is not None),
    )
    return x


class MQALayer(nn.Module):
    """Multi-head Latent Attention (MLA) with sliding window + optional KV compression.
     Uses low-rank Q projection (wq_a -> q_norm -> wq_b) and grouped low-rank O projection.
     架构概述 (以输入 x: [b, s, 4096] 为例, 默认单卡 world_size=1):
     ┌──────────────────────────────────────────────────────────────────────┐
     │ Q路径:  x[b,s,4096] -> wq_a -> q_norm -> wq_b -> Q[b,s,64,512]   │
     │ KV路径: x[b,s,4096] -> wkv -> kv_norm -> KV[b,s,512] (共享latent) │
     │ O路径:  attn_out[b,s,64,512] -> wo_a(分组低秩) -> wo_b -> [b,s,4096]│
     └──────────────────────────────────────────────────────────────────────┘
     关键参数: dim=4096, n_heads=64, head_dim=512, rope_head_dim=64,
               q_lora_rank=1024, o_lora_rank=1024, o_groups=8, window_size=128
     """
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_rank = attn_tp_rank = get_attention_tp_rank()
        self.tp_size = attn_tp_size = get_attention_tp_size()
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_cp_size()
            self.tp_rank = attn_tp_rank = 0
            self.tp_size = attn_tp_size = 1
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // attn_tp_size
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // attn_tp_size
        self.rope_head_dim = config.qk_rope_head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        compress_ratio = (
            compress_ratio_override
            if compress_ratio_override is not None
            else config.compress_ratios[layer_id]
        )
        assert compress_ratio in [0, 4, 128]
        self.compress_ratio: Literal[0, 4, 128] = compress_ratio

        assert self.head_dim == config.head_dim
        assert config.num_key_value_heads == 1

        rope_theta, rope_scaling = get_rope_config(config)
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        rope_base = config.compress_rope_theta if self.compress_ratio else rope_theta

        from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis

        assert self.compress_ratio in {0, 4, 128}
        if self.compress_ratio:
            original_seq_len = rope_scaling["original_max_position_embeddings"]
        else:
            original_seq_len = 0

        freqs_cis = precompute_freqs_cis(
            dim=self.qk_rope_head_dim,
            seqlen=config.max_position_embeddings,
            original_seq_len=original_seq_len,
            base=rope_base,
            factor=rope_scaling["factor"],
            beta_fast=rope_scaling["beta_fast"],
            beta_slow=rope_scaling["beta_slow"],
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.freqs_cis: torch.Tensor

        if envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() and alt_streams is not None:
            self.alt_streams = alt_streams[:3]
            self.alt_streams_indexer = alt_streams[-2:]
        else:
            self.alt_streams = None
            self.alt_streams_indexer = None

        from sglang.srt.utils import is_blackwell_supported

        self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64

        self.compressor = None
        self.indexer = None
        # ====== KV压缩器与索引器 (仅在compress_ratio>0时启用) ======
        if self.compress_ratio:
            self.compressor = Compressor(
                config,
                layer_id=self.layer_id,
                is_in_indexer=False,
                freqs_cis=freqs_cis,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                rotate=False,
                prefix=add_prefix("compressor", prefix),
            )
            if self.compress_ratio == 4:
                # 压缩比为4时, 使用Indexer进行稀疏注意力索引 (学习式top-k选择)
                self.indexer = C4Indexer(
                    config,
                    freqs_cis=freqs_cis,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix("indexer", prefix),
                    alt_streams=self.alt_streams_indexer,
                )
            # 压缩比=128时, 仅使用规则式索引, 无需Indexer
        # ====== 注意力汇聚参数 ======
        # attn_sink: 可学习的注意力偏置, 用于吸收注意力分数中的"汇聚"效应
        # weight shape: [64] (float32), 每个头一个标量
        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.fuse_wqa_wkv = envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        if self.fuse_wqa_wkv:
            self.wqkv_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wqkv_a", prefix),
            )
        else:
            # ====== Q投影: 两阶段低秩投影 (MLA核心) ======
            # wq_a: Q的降维投影 (dim -> q_lora_rank)
            #   weight shape: [1024, 4096] (float8_e4m3fn)
            #   scale shape:  [8, 32]      (float8_e8m0fnu, 每128元素一组: 1024/128=8, 4096/128=32)
            self.wq_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wq_a", prefix),
            )
            # ====== KV投影: MLA共享潜在表示 (所有头共享一个latent) ======
            # wkv: 将输入映射为单个共享KV latent (dim -> head_dim)
            #   weight shape: [512, 4096] (float8_e4m3fn)
            #   scale shape:  [4, 32]     (float8_e8m0fnu, 512/128=4, 4096/128=32)
            self.wkv = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wkv", prefix),
            )
        # q_norm: Q降维后的RMSNorm
        #   weight shape: [1024] (float32)
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        # wq_b: Q的升维投影 (q_lora_rank -> n_heads * head_dim), 按头维度切分(ColumnParallel)
        #   weight shape: [32768, 1024] (float8_e4m3fn, 单卡: 64头×512维=32768)
        #   scale shape:  [256, 8]      (float8_e8m0fnu, 32768/128=256, 1024/128=8)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        # kv_norm: KV latent的RMSNorm
        #   weight shape: [512] (float32)
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)
        # ====== O投影: 分组低秩投影 ======
        # wo_a: O的降维投影, 按分组进行 (每组: n_heads*head_dim/n_groups -> o_lora_rank)
        #   实际为 ColumnParallelLinear(4096, 8192, dtype=bfloat16)
        #   weight shape: [8192, 4096] (bfloat16, 8组×1024秩=8192)
        #   无scale (bfloat16不需要量化)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config if _FP8_WO_A_GEMM else None,
            prefix=add_prefix("wo_a", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            **({} if _FP8_WO_A_GEMM else {"params_dtype": torch.bfloat16}),
        )
        if _FP8_WO_A_GEMM:
            assert hasattr(
                self.wo_a, "weight_scale_inv"
            ), "FP8 quant_config must create weight_scale_inv"
            self.wo_a.weight_scale_inv.format_ue8m0 = True
        # wo_b: O的升维投影, 将分组低秩结果合并回dim (n_groups*o_lora_rank -> dim)
        #   实际为 RowParallelLinear(8192, 4096)
        #   weight shape: [4096, 8192] (float8_e4m3fn)
        #   scale shape:  [32, 64]     (float8_e8m0fnu, 4096/128=32, 8192/128=64)
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=attn_tp_size > 1,
            prefix=add_prefix("wo_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.attn_mqa = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.softmax_scale,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        # KV cache write is always fused into the K kernel
        # (`_compute_kv_to_cache`), so the legacy "overlap store cache" flag
        # has no effect here -- the fused path is on by default.

    def _compute_q_a(
        self,
        x: torch.Tensor,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if qkv_a is not None:
            q = qkv_a[..., : self.q_lora_rank]
        else:
            q, _ = self.wq_a(x)
        return self.q_norm(q)

    def _compute_q_b(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        q_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if q_out is None:
            q_out = torch.empty_like(q)
        # Fused warp-per-(token, head) rmsnorm-self + RoPE + write to q_out.
        fused_q_norm_rope(q, q_out, self.eps, self.freqs_cis, positions)
        return q_out

    def _compute_kv_to_cache(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> None:
        """Fused: rmsnorm + RoPE + write directly to FlashMLA paged cache.

        Replaces the bf16-kv-intermediate path. Used everywhere except the NSA
        prefill-CP case (which needs bf16 kv for the cross-rank all-gather).
        """
        if qkv_a is not None:
            kv = qkv_a[..., self.q_lora_rank :]
        else:
            kv, _ = self.wkv(x)
        token_to_kv_pool = forward_batch.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        token_to_kv_pool.set_swa_key_buffer_radix_fused_norm_rope(
            layer_id=self.layer_id,
            raw_loc=forward_batch.out_cache_loc,
            kv=kv,
            kv_weight=self.kv_norm.weight.data,
            eps=self.eps,
            freqs_cis=self.freqs_cis,
            positions=positions,
        )

    def _compute_kv_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bf16-kv path used by the NSA prefill-CP case (needs all-gather)."""
        if qkv_a is not None:
            kv = qkv_a[..., self.q_lora_rank :]
        else:
            kv, _ = self.wkv(x)
        kv = kv.contiguous()
        fused_norm_rope_inplace(
            kv,
            self.kv_norm.weight.data,
            self.eps,
            self.freqs_cis,
            positions,
        )
        return kv

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: DeepseekV4AttnBackend,
        q_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self.alt_streams is not None
        assert len(self.alt_streams) >= 3

        current_stream = torch.cuda.current_stream()
        stream_kv = self.alt_streams[0]
        stream_compressor = self.alt_streams[1]
        stream_indexer = self.alt_streams[2]

        stream_kv.wait_stream(current_stream)
        stream_compressor.wait_stream(current_stream)
        stream_indexer.wait_stream(current_stream)

        qkv_a: Optional[torch.Tensor] = None
        qkv_a_ready: Optional[torch.cuda.Event] = None
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x)
            qkv_a_ready = current_stream.record_event()

        q_lora = self._compute_q_a(x, qkv_a=qkv_a)
        q_lora_ready = current_stream.record_event()

        if self.indexer is not None:
            with torch.cuda.stream(stream_indexer):
                self.indexer(
                    x=x,
                    q_lora=q_lora,
                    forward_batch=forward_batch,
                    enable_multi_stream=True,
                    q_lora_ready=q_lora_ready,
                )

        with torch.cuda.stream(stream_kv):
            if qkv_a_ready is not None:
                stream_kv.wait_event(qkv_a_ready)
            # Fused norm + rope + cache write -- no bf16 KV intermediate.
            self._compute_kv_to_cache(x, positions, forward_batch, qkv_a=qkv_a)

        del qkv_a

        if self.compressor is not None:
            with torch.cuda.stream(stream_compressor):
                attn_backend.forward_core_compressor(
                    x, forward_batch, self.layer_id, self.compressor
                )

        q = self._compute_q_b(q_lora, positions, q_out)
        current_stream.wait_stream(stream_kv)
        current_stream.wait_stream(stream_compressor)
        current_stream.wait_stream(stream_indexer)

        return q

    def _forward_prepare(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: DeepseekV4AttnBackend,
        q_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x)
            q_lora = qkv_a[..., : self.q_lora_rank]
        else:
            q_lora, _ = self.wq_a(x)
            qkv_a = None
        q_lora = self.q_norm(q_lora)
        q = self._compute_q_b(q_lora, positions, q_out)

        use_cp = self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch)
        kv: Optional[torch.Tensor]
        if use_cp:
            # NSA CP: keep bf16 kv around for the cross-rank all-gather, then
            # write to the FlashMLA cache after gather.
            kv = self._compute_kv_bf16(x, positions, qkv_a=qkv_a)
            kv = cp_all_gather_rerange_output(
                kv.contiguous(),
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
            attn_backend.store_cache(
                layer_id=self.layer_id,
                swa_k=kv,
                forward_batch=forward_batch,
            )
        else:
            self._compute_kv_to_cache(x, positions, forward_batch, qkv_a=qkv_a)
            kv = None

        del qkv_a

        if self.indexer is not None:
            self.indexer(x=x, q_lora=q_lora, forward_batch=forward_batch)
        if self.compressor is not None:
            attn_backend.forward_core_compressor(
                x,
                forward_batch,
                self.layer_id,
                self.compressor,
            )

        return q, kv

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if not get_attn_tp_context().input_scattered and x.shape[0] == 0:
            assert (
                not self.wo_b.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return x

        attn_backend = forward_batch.attn_backend
        if TYPE_CHECKING:
            assert isinstance(attn_backend, DeepseekV4AttnBackend)

        enable_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and self.alt_streams is not None
            and get_is_capture_mode()
            and x.shape[0] <= self._multi_stream_bs_limit
            and not (self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch))
        )

        tp_slice, q_padded, q_out = slice(None), None, None
        if self.tp_size > 1:
            q_padded = x.new_empty(x.shape[0], self.n_heads, self.head_dim)
            rank = self.tp_rank
            tp_slice = slice(rank * self.n_local_heads, (rank + 1) * self.n_local_heads)
            q_out = q_padded[:, tp_slice, :]

        if enable_multi_stream:
            # Multi-stream path always fuses cache write into the K kernel,
            # so the bf16 KV intermediate is gone.
            q = self._forward_prepare_multi_stream(
                x, positions, forward_batch, attn_backend, q_out
            )
            kv = None
        else:
            q, kv = self._forward_prepare(
                x, positions, forward_batch, attn_backend, q_out
            )

        # The cache write is always fused / already done by _forward_prepare* --
        # tell the backend to skip its own store_cache. When `kv is None`
        # (no NSA-CP), pass `q` as a sentinel for the `k is v` assert; the
        # attention path doesn't read it once `save_kv_cache=False`.
        attn_k = kv if kv is not None else q
        o = attn_backend.forward(
            q=q_padded if q_padded is not None else q,
            k=attn_k,
            v=attn_k,
            layer=self.attn_mqa,
            forward_batch=forward_batch,
            compress_ratio=self.compress_ratio,
            attn_sink=self.attn_sink,
            save_kv_cache=False,
        )
        o = o[:, tp_slice, :]
        fused_rope_inplace(
            o[..., -self.qk_rope_head_dim :],
            None,
            self.freqs_cis,
            positions=positions,
            inverse=True,
        )

        o = o.view(o.shape[0], self.n_local_groups, -1)

        if _FP8_WO_A_GEMM:
            import deep_gemm

            T, G, D = o.shape
            R = self.o_lora_rank
            o_fp8, o_s = sglang_per_token_group_quant_fp8(
                o.reshape(T * G, D).contiguous(),
                group_size=128,
            )
            o_s = deep_gemm.ceil_to_ue8m0(o_s)
            output = torch.empty(T, G, R, device=o.device, dtype=torch.bfloat16)
            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd",
                (o_fp8.view(T, G, D), o_s.view(T, G, -1)),
                (self.wo_a.weight.view(G, R, D), self.wo_a.weight_scale_inv.data),
                output,
                recipe=(1, 1, 128),
            )
            o = output
        else:
            wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            o = torch.einsum("tgd,grd->tgr", o, wo_a)

        o, _ = self.wo_b(o.flatten(1))

        return o


class DeepseekV4DecoderLayer(nn.Module):
    """Transformer block with Hyper-Connections (HC) mixing.
    Instead of a simple residual, HC maintains `hc_mult` copies of the hidden state.
    hc_pre: reduces hc copies -> 1 via learned weighted sum (pre-weights from Sinkhorn).
    hc_post: expands 1 -> hc copies via learned post-weights + combination matrix.
    Hyper-Connection (HC) 架构概述 (输入 x: [B, S, 4, 4096]):
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ hc_pre:  [B,S,4,4096] ──→ (Sinkhorn加权求和) ──→ [B,S,4096]            │
    │          产出 pre:[B,S,4], post:[B,S,4], comb:[B,S,4,4]                  │
    │                                                                          │
    │ Attn/FFN: [B,S,4096] ──→ norm ──→ attn/ffn ──→ [B,S,4096]              │
    │                                                                          │
    │ hc_post: [B,S,4096] + residual[B,S,4,4096] ──→ (加权扩展) ──→ [B,S,4,4096]│
    └────────────────────────────────────────────────────────────────────────────┘
    关键参数: hc_mult=4, dim=4096, mix_hc=(2+4)*4=24, hc_dim=4*4096=16384
    """
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.self_attn = MQALayer(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_streams=alt_streams,
            compress_ratio_override=compress_ratio_override,
        )
        self.mlp = deepseek_v2.DeepseekV2MoE(
            config=config,
            quant_config=moe_quant_config_override or quant_config,
            prefix=add_prefix("mlp", prefix),
            layer_id=self.layer_id,
            alt_stream=alt_streams[0] if alt_streams is not None else None,
            is_nextn=is_nextn,
            is_deepseek_v4=True,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hc_mult = hc_mult = config.hc_mult  # 4 - HC副本数
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters   # 20 - Sinkhorn迭代次数
        self.hc_eps = config.hc_eps  # 1e-6 - Sinkhorn的epsilon
        # ====== Hyper-Connection (HC) 参数 ======
        # mix_hc = (2 + hc_mult) * hc_mult = (2+4)*4 = 24
        # 这个24维向量被拆分为三部分: pre(4) + post(4) + comb(4×4=16)
        mix_hc = (2 + hc_mult) * hc_mult
        # hc_dim = hc_mult * dim = 4 * 4096 = 16384
        hc_dim = hc_mult * config.hidden_size
        # hc_attn_fn: Attention HC的混合函数权重
        #   todo shape: [24, 16384] (float32)
        #   输入: [B, S, 16384] (4个HC副本flatten), 输出: [B, S, 24] (混合系数)
        #   24 = pre(4) + post(4) + comb(4×4=16)
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        # 同上结构
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        # hc_attn_base: Attention HC的偏置项 (用于sigmoid变换)
        #   todo shape: [24] (float32)
        #   与hc_scale一起来计算【pre、post、comb】:
        #   pre = sigmoid(mixes[:4] * scale[0] + base[:4])
        #   post = 2*sigmoid(mixes[4:8] * scale[1] + base[4:8])
        #   comb = Sinkhorn(mixes[8:24] * scale[2] + base[8:24])
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        # 同上结构
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        # hc_attn_scale: Attention HC的缩放因子 (3个标量, 分别对应pre/post/comb)
        #   todo shape: [3] (float32)
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        # 同上结构
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.rms_norm_eps = config.rms_norm_eps
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()

    def prewarm_mhc_token_counts(
        self, token_counts: Tuple[int, ...], device: torch.device
    ) -> None:
        paths = (
            (
                "attn",
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.input_layernorm,
            ),
            (
                "ffn",
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                self.post_attention_layernorm,
            ),
        )

        with torch.inference_mode():
            for num_tokens in token_counts:
                for path_name, hc_fn, hc_scale, hc_base, norm in paths:
                    tic = time.perf_counter()
                    residual = torch.empty(
                        (num_tokens, self.hc_mult, self.hidden_size),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    y, post, comb, _ = self.hc_pre(
                        residual,
                        hc_fn,
                        hc_scale,
                        hc_base,
                        norm=norm,
                    )
                    del residual, y, post, comb
                    torch.cuda.synchronize()
                    logger.info(
                        "DeepSeek V4 MHC prewarm path=%s num_tokens=%s completed in %.3fs",
                        path_name,
                        num_tokens,
                        time.perf_counter() - tic,
                    )

    def prewarm_mhc_token_count_buckets(
        self, max_num_tokens: int, device: torch.device
    ) -> Tuple[int, ...]:
        from sglang.srt.layers.mhc import get_mhc_pre_token_count_representatives

        token_counts = get_mhc_pre_token_count_representatives(
            max_num_tokens, self.hc_mult * self.hidden_size
        )
        if not token_counts:
            return token_counts

        logger.info(
            "DeepSeek V4 MHC prewarm max_num_tokens=%s representative token counts: %s",
            max_num_tokens,
            token_counts,
        )
        self.prewarm_mhc_token_counts(token_counts, device)
        return token_counts

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: Optional[nn.Module] = None,
    ):
        """If *norm* is given and the TileLang path is active, the returned
        hidden_states are already post-norm (the norm is fused into the kernel)."""
        """
                Hyper-Connection 前处理: 将 hc_mult=4 个HC副本加权合并为1个隐藏状态
                核心逻辑:
                1. 对4个HC副本flatten后做线性投影 → 得到24维混合系数mixes
                2. 通过Sinkhorn归一化将mixes分解为: pre权重[4], post权重[4], 组合矩阵[4,4]
                3. 用pre权重对4个HC副本做加权求和 → 输出单一隐藏状态
                参数 Shape:
                  x:        [B, S, 4, 4096]  - 4个HC副本的隐藏状态
                  hc_fn:    [24, 16384]      - 混合函数权重 (float32)
                  hc_scale: [3]              - pre/post/comb的缩放因子 (float32)
                  hc_base:  [24]             - pre/post/comb的偏置 (float32)
                返回:
                  y:    [B, S, 4096]         - 合并后的单一隐藏状态
                  post: [B, S, 4]            - post-weights (给hc_post用)
                  comb: [B, S, 4, 4]         - 组合矩阵 (给hc_post用)
                """

        @compile_in_capture_mode
        def hc_pre_torch_impl(x, hc_fn):
            # Step 1: Flatten HC维度 + 转float32
            #   x: [B, S, 4, 4096] → x.flatten(2): [B, S, 16384]
            #   .float(): 转为float32 (HC计算需要高精度)
            x_flat = x.flatten(1).float()
            # Step 2: RMS归一化因子 (对flatten后的16384维做归一化)
            #   x.square(): [B, S, 16384]
            #   .mean(-1, keepdim=True): [B, S, 1] - 每个token的均方值
            #   rsqrt: [B, S, 1] - 逆平方根归一化因子
            rsqrt = torch.rsqrt(
                x_flat.square().mean(-1, keepdim=True) + self.rms_norm_eps
            )
            # todo mixes
            # rsqrt: [B, S, 1] (fp32)
            # Step 3: 线性投影 + RMS归一化 → 得到混合系数
            #   F.linear(x, hc_fn): [B, S, 16384] × [24, 16384]^T → [B, S, 24]
            #   * rsqrt: [B, S, 24] × [B, S, 1] → [B, S, 24]
            #   24维被拆分为: pre[:4] + post[4:8] + comb[8:24](即4×4矩阵)
            mixes = (F.linear(x_flat, hc_fn) * rsqrt).unsqueeze(1)
            return x_flat, mixes

        shape, dtype = x.size(), x.dtype

        if x.shape[0] == 0:
            y = torch.empty((0, shape[-1]), dtype=dtype, device=x.device)
            post = torch.empty((0, self.hc_mult), dtype=dtype, device=x.device)
            comb = torch.empty(
                (0, self.hc_mult, self.hc_mult), dtype=dtype, device=x.device
            )
            return y, post, comb, False

        if envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            from sglang.srt.layers.mhc import mhc_pre

            norm_kwargs = {}
            if norm is not None:
                norm_kwargs["norm_weight"] = norm.weight.data
                norm_kwargs["norm_eps"] = norm.variance_epsilon

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
                **norm_kwargs,
            )
            return y, post.squeeze(-1), comb, norm is not None

        if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
            import deep_gemm

            x_flat = x.flatten(1).bfloat16()

            m, k = x_flat.shape
            mix_hc = hc_fn.size(0)
            d_out = torch.empty((m, mix_hc), dtype=torch.float, device=x.device)
            s_out = torch.empty((m,), dtype=torch.float, device=x.device)
            deep_gemm.tf32_hc_prenorm_gemm(
                x_flat, hc_fn.float().contiguous(), d_out, s_out, num_splits=None
            )
            rsqrt = torch.rsqrt(s_out / k + self.rms_norm_eps)
            mixes = (d_out * rsqrt.unsqueeze(1)).unsqueeze(1)
        else:
            # todo
            x_flat, mixes = hc_pre_torch_impl(x, hc_fn)

        from sglang.srt.layers.mhc import hc_split_sinkhorn
        # Step 4: Sinkhorn归一化 - 将mixes分解为pre, post, comb三个部分
        #   输入:
        #     mixes:    [B*S, 24]      - 混合系数 (view为2D传入kernel)
        #     hc_scale: [3]            - 缩放因子 [pre_scale, post_scale, comb_scale]
        #     hc_base:  [24]           - 偏置项
        #   内部计算:
        #     pre[i]  = sigmoid(mixes[i,:4] * scale[0] + base[:4]) + eps    → [B*S, 4]
        #     post[i] = 2*sigmoid(mixes[i,4:8] * scale[1] + base[4:8])     → [B*S, 4]
        #     comb    = Sinkhorn(mixes[i,8:24] * scale[2] + base[8:24])     → [B*S, 4, 4]
        #     Sinkhorn: 先softmax(dim=-1)+eps, 再交替归一化行和列20次, 得到双随机矩阵
        #   输出:
        #     pre:  [B, S, 4]    - 前向权重 (每个HC副本的贡献权重, sigmoid>0)
        #     post: [B, S, 4]    - 后向权重 (hc_post中用于扩展, 2*sigmoid范围[0,2])
        #     comb: [B, S, 4, 4] - 组合矩阵 (双随机矩阵, 行和≈1, 列和≈1)
        pre, post, comb = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        # Step 5: 用pre权重对4个HC副本做加权求和 → 合并为单一隐藏状态
        #   pre.unsqueeze(-1): [B, S, 4, 1]    - 权重维度
        #   x.view(shape):     [B, S, 4, 4096] - 恢复4个HC副本的原始shape (float32)
        #   相乘: [B, S, 4, 1] × [B, S, 4, 4096] → [B, S, 4, 4096] (广播)
        #   torch.sum(dim=2):  [B, S, 4, 4096] → [B, S, 4096] (沿HC维度求和)
        y = (pre.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(dim=1)
        # todo squeeze(1) 用于移除张量中第 1 维（即第二个维度，索引从 0 开始），但仅当该维度的大小为 1 时才会移除；
        #  如果该维度大小不为 1，则张量保持不变。
        return y.to(dtype), post.squeeze(1), comb.squeeze(1), False

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        """
             Hyper-Connection 后处理: 将Attn/FFN的单一输出扩展回 hc_mult=4 个HC副本,
             并与残差(residual)通过组合矩阵(comb)混合
             核心逻辑:
             1. post权重: 将Attn/FFN输出复制4份, 各乘以不同post权重
             2. comb矩阵: 将4个残差副本通过4×4双随机矩阵交叉混合
             3. 两者相加 → 新的4个HC副本
             参数 Shape:
               x:        [B, S, 4096]    - Attn/FFN的输出 (单一隐藏状态)
               residual: [B, S, 4, 4096] - HC残差 (hc_pre前的4个HC副本)
               post:     [B, S, 4]       - post-weights (来自hc_pre)
               comb:     [B, S, 4, 4]    - 组合矩阵 (来自hc_pre, 双随机矩阵)
             返回:
               y: [B, S, 4, 4096]       - 新的4个HC副本
             """

        if x.shape[0] == 0:
            return torch.empty(
                (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
            )

        if envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get():
            from sglang.srt.layers.mhc import mhc_post

            return mhc_post(x, residual, post, comb)

        assert residual.shape == (x.shape[0], self.hc_mult, x.shape[-1])
        assert post.shape == (x.shape[0], self.hc_mult)
        assert comb.shape == (x.shape[0], self.hc_mult, self.hc_mult)

        @compile_in_capture_mode
        def hc_post_torch_impl(x, residual, post, comb):
            # 第一项: post权重 × Attn/FFN输出 (扩展为4个副本)
            #   post.unsqueeze(-1):  [B, S, 4, 1]     - post权重
            #   x.unsqueeze(-2):     [B, S, 1, 4096]   - Attn/FFN输出 (插入HC维度)
            #   相乘: [B, S, 4, 1] × [B, S, 1, 4096] → [B, S, 4, 4096] (广播)
            #   效果: 每个HC副本 = post[i] × x, i=0..3
            #   post范围: [0, 2] (2*sigmoid), 不同副本获得不同缩放的Attn/FFN输出
            # 第二项: comb矩阵 × 残差 (HC副本间的交叉混合)
            #   comb.unsqueeze(-1):    [B, S, 4, 4, 1]    - 组合矩阵 (双随机)
            #   residual.unsqueeze(-2): [B, S, 1, 4, 4096] - 残差 (插入comb输出维度)
            #   相乘: [B, S, 4, 4, 1] × [B, S, 1, 4, 4096] → [B, S, 4, 4, 4096] (广播)
            #   torch.sum(dim=2):      [B, S, 4, 4, 4096] → [B, S, 4, 4096] (沿源HC维度求和)
            #   效果: 新HC副本[i] = Σ_j comb[i,j] * residual[j], 即4个旧副本的加权组合
            return (
                post.unsqueeze(-1) * x.unsqueeze(1)
                + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
            ).type_as(x) # [B, S, 4, 4096] (bf16)

        return hc_post_torch_impl(x, residual, post, comb)

    def forward(
        self,
        positions: torch.tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids_global: torch.Tensor,
    ) -> torch.Tensor:
        """
                Block前向传播 - 输入 x: [B, S, 4, 4096], 4个HC副本
                数据流:
                x[B,S,4,4096] ──→ hc_pre ──→ [B,S,4096] ──→ attn_norm ──→ attn ──→ [B,S,4096]
                  │                                                  │
                  └──── residual[B,S,4,4096] ←──────────────────────┘
                                    │
                            hc_post(attn_out, residual, post, comb) ──→ [B,S,4,4096]
                                    │
                  ┌─────────────────┘
                  ↓
                [B,S,4,4096] ──→ hc_pre ──→ [B,S,4096] ──→ ffn_norm ──→ ffn ──→ [B,S,4096]
                  │                                                  │
                  └──── residual[B,S,4,4096] ←──────────────────────┘
                                    │
                            hc_post(ffn_out, residual, post, comb) ──→ [B,S,4,4096]
                """
        residual = hidden_states # residual: [B, S, 4, 4096]
        # todo hc_pre 做两件事：
        # todo 1. 用 pre 权重把 [T, 4, D] 压成 [T, D]
        # todo 2. 同时生成 post 和 comb，留给 attention 后的 hc_post 用
        # todo hidden_states就是pre
        # todo 4====>1
        # hc_pre: 4个HC副本 → 1个隐藏状态
        #   输入: hidden_states[B,S,4,4096], hc_attn_fn[24,16384], hc_attn_scale[3], hc_attn_base[24]
        #   输出: hidden_states:[B,S,4096], post:[B,S,4], comb:[B,S,4,4]
        hidden_states, post, comb, norm_fused = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            norm=self.input_layernorm,
        )
        if not norm_fused:
            # attn_norm: RMSNorm
            #   输入: [B, S, 4096], weight: [4096]
            #   输出: [B, S, 4096]
            hidden_states = self.input_layernorm(hidden_states)
        # attn: 注意力计算
        #   输入: [B, S, 4096], 输出: [B, S, 4096]
        hidden_states = self.self_attn(
            x=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
        )
        # todo [B, S, 4, 4096] (bf16)
        # todo 1====>4
        # hc_post: Attn输出 → 4个HC副本 (与残差混合)
        #   输入: hidden_states[B,S,4096], residual[B,S,4,4096], post[B,S,4], comb[B,S,4,4]
        #   输出: hidden_states[B, S, 4, 4096]
        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        residual = hidden_states
        # hc_pre: 4个HC副本 → 1个隐藏状态
        #   输入: hidden_states[B,S,4,4096], hc_ffn_fn[24,16384], hc_ffn_scale[3], hc_ffn_base[24]
        #   输出: hidden_states:[B,S,4096], post:[B,S,4], comb:[B,S,4,4]
        hidden_states, post, comb, norm_fused = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm=self.post_attention_layernorm,
        )
        if not norm_fused:
            # ffn_norm: RMSNorm
            #   输入: [B, S, 4096], weight: [4096]
            #   输出: [B, S, 4096]
            hidden_states = self.post_attention_layernorm(hidden_states)

        _use_cp = self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch)
        _use_tp_moe_gather = (
            not _use_cp
            and get_attention_dp_size() > 1
            and get_moe_a2a_backend().is_none()
        )
        _use_tp_attn_a2a_scatter = (
            not _use_cp
            and envs.SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER.get()
            and get_attention_tp_size() > 1
            and not get_moe_a2a_backend().is_none()
        )
        if _use_cp:
            assert get_moe_a2a_backend().is_deepep(), (
                "CP requires DeepEP (moe_a2a_backend == deepep). "
                "Only DeepEP is tested with CP's per-rank token split."
            )
            cp_rank = get_attention_cp_rank()
            cp_size = get_attention_cp_size()
            input_ids = input_ids[cp_rank::cp_size].contiguous()
            input_ids_global = input_ids
        elif _use_tp_moe_gather:
            hidden_states, local_hidden_states = (
                get_global_dp_buffer(get_tp_group()),
                hidden_states,
            )
            dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
        _a2a_scatter_chunks: Optional[List[torch.Tensor]] = None
        if _use_tp_attn_a2a_scatter:
            s, r = get_attention_tp_size(), get_attention_tp_rank()
            _a2a_scatter_chunks = list(hidden_states.tensor_split(s))
            hidden_states = _a2a_scatter_chunks[r].contiguous()
            input_ids = input_ids.tensor_split(s)[r].contiguous()
            input_ids_global = input_ids_global.tensor_split(s)[r].contiguous()
        # ffn: MoE前馈网络
        #   输入: [B, S, 4096], 输出: [B, S, 4096]
        hidden_states = self.mlp(
            hidden_states,
            forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
        )
        if _use_tp_moe_gather:
            hidden_states, global_hidden_states = (
                get_local_dp_buffer(get_tp_group()),
                hidden_states,
            )
            dp_scatter(hidden_states, global_hidden_states, forward_batch)
        if _use_tp_attn_a2a_scatter:
            assert _a2a_scatter_chunks is not None
            gathered = [torch.empty_like(t) for t in _a2a_scatter_chunks]
            attn_tp_all_gather(gathered, hidden_states.contiguous())
            hidden_states = torch.cat(gathered)
        # hc_post: FFN输出 → 4个HC副本 (与残差混合)
        #   输入: hidden_states[B,S,4096], residual[B,S,4,4096], post[B,S,4], comb[B,S,4,4]
        #   输出: hidden_states[B, S, 4, 4096]
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        return hidden_states


class DeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.hidden_size = config.hidden_size
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.rms_norm_eps = config.rms_norm_eps
        self.alt_streams = (
            [torch.cuda.Stream() for _ in range(5)] if (_is_cuda or _is_hip) else None
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: DeepseekV4DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_streams=self.alt_streams,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.gemm_output_zero_allocator_size = 0
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        if self.pp_group.is_last_rank:
            hc_dim = hc_mult * config.hidden_size
            self.hc_head_fn = nn.Parameter(
                torch.empty(hc_mult, hc_dim, dtype=torch.float32)
            )
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_cp_size()

    def prewarm_mhc_token_count_buckets(
        self, max_num_tokens: int, device: torch.device
    ) -> Tuple[int, ...]:
        tic = time.perf_counter()
        logger.info(
            "Running DeepSeek V4 MHC prewarm for max_num_tokens=%s",
            max_num_tokens,
        )
        token_counts = self.layers[self.start_layer].prewarm_mhc_token_count_buckets(
            max_num_tokens, device
        )
        logger.info(
            "DeepSeek V4 MHC prewarm finished in %.3fs for representative token counts: %s",
            time.perf_counter() - tic,
            token_counts,
        )
        return token_counts

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        """
             HC头: 将4个HC副本加权合并为单一隐藏状态 (简化版hc_pre, 无Sinkhorn)
             与Block.hc_pre的区别:
             - hc_pre使用Sinkhorn归一化, 产出pre+post+comb (需要hc_post逆变换)
             - hc_head仅使用sigmoid, 产出pre (只需要合并, 无需逆变换)
             - hc_fn shape: [4, 16384] (只需4维输出, 而非24维)
             - hc_scale shape: [1] (1个标量, 而非3个)
             参数 Shape:
               x:        [B, S, 4, 4096]  - 4个HC副本的隐藏状态
               hc_fn:    [4, 16384]       - 混合函数权重 (float32)
               hc_scale: [1]              - 缩放因子 (float32)
               hc_base:  [4]              - 偏置项 (float32)
             返回:
               y: [B, S, 4096] - 合并后的单一隐藏状态
             """
        if x.numel() > 0:
            from sglang.srt.layers.mhc_head import fused_hc_head

            return fused_hc_head(
                x.contiguous(),
                hc_fn,
                hc_scale,
                hc_base,
                norm_eps=self.norm_eps,
                hc_eps=self.hc_eps,
            )
        # shape=[B,S,4,4096], dtype=bf16
        shape, dtype = x.size(), x.dtype
        # Step 1: Flatten HC维度 + 转float32
        #   x: [B, S, 4, 4096] → x.flatten(2): [B, S, 16384] (fp32)
        x = x.flatten(1).float()
        # Step 2: RMS归一化因子
        #   x.square().mean(-1, keepdim=True): [B, S, 1] - 每个token的均方值
        #   rsqrt: [B, S, 1] - 逆平方根
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        # rsqrt: [B, S, 1] (fp32)
        # Step 3: 线性投影 + RMS归一化 → 得到混合系数
        #   F.linear(x, hc_fn): [B, S, 16384] × [4, 16384]^T → [B, S, 4]
        #   * rsqrt: [B, S, 4] × [B, S, 1] → [B, S, 4]
        mixes = F.linear(x, hc_fn) * rsqrt
        # Step 4: sigmoid变换 → pre权重 (每个HC副本的贡献权重)
        #   mixes * hc_scale: [B, S, 4] × [1] → [B, S, 4] (缩放)
        #   + hc_base: [B, S, 4] + [4] → [B, S, 4] (偏置)
        #   sigmoid: [B, S, 4] → [B, S, 4] (值域[0,1])
        #   + hc_eps: [B, S, 4] → [B, S, 4] (加小量避免零权重, 值域[eps,1+eps])
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        # pre: [B, S, 4] (fp32)
        # Step 5: 用pre权重对4个HC副本做加权求和 → 合并为单一隐藏状态
        #   pre.unsqueeze(-1): [B, S, 4, 1]    - 权重维度
        #   x.view(shape):     [B, S, 4, 4096] - 恢复4个HC副本 (fp32)
        #   相乘: [B, S, 4, 1] × [B, S, 4, 4096] → [B, S, 4, 4096] (广播)
        #   torch.sum(dim=2):  [B, S, 4, 4096] → [B, S, 4096] (沿HC维度求和)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            hidden_states = self.embed_tokens(input_ids)
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            # Unflatten 2D PP IPC tensor back to 3D mHC shape.
            if hidden_states.ndim == 2:
                hidden_states = hidden_states.view(
                    hidden_states.shape[0], self.hc_mult, self.hidden_size
                )

        if get_attention_dp_size() > 1 and get_moe_a2a_backend().is_none():
            input_ids_global = torch.empty(
                (_DpGatheredBufferWrapper._global_dp_buffer_len, 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            dp_gather_partial(input_ids_global, input_ids[:, None], forward_batch)
            input_ids_global = input_ids_global.squeeze(-1)
        else:
            input_ids_global = input_ids

        if nsa_use_prefill_cp(forward_batch):
            if self.pp_group.is_first_rank:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                input_ids=input_ids,
                input_ids_global=input_ids_global,
            )

        # CP all-gather only on the last PP rank; PP IPC carries CP-split tensors.
        if self.pp_group.is_last_rank and nsa_use_prefill_cp(forward_batch):
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )

        if not self.pp_group.is_last_rank:
            # Flatten 3D mHC tensor for PP IPC.
            return PPProxyTensors({"hidden_states": hidden_states.flatten(1)})

        pre_hc_head = hidden_states.flatten(1)
        # todo
        hidden_states = self.hc_head(
            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        hidden_states = self.norm(hidden_states)

        return hidden_states, pre_hc_head


class DeepseekV4ForCausalLM(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.model = DeepseekV4Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.pp_group = get_pp_group()
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        get_attn_tp_context().init_context(config.q_lora_rank, is_nsa=True)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.model.start_layer, self.model.end_layer)
                if isinstance(
                    self.model.layers[layer_id].mlp, deepseek_v2.DeepseekV2MoE
                )
            }
        )

        # Expose start_layer/end_layer for model_runner PP support
        self.start_layer = self.model.start_layer
        self.end_layer = self.model.end_layer

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_rank = get_attention_cp_rank()
            self.cp_size = get_attention_cp_size()

    def prewarm_mhc_token_count_buckets(
        self, max_num_tokens: int, device: torch.device
    ) -> Tuple[int, ...]:
        return self.model.prewarm_mhc_token_count_buckets(max_num_tokens, device)

    def kernel_warmup(self, model_runner) -> None:
        if not model_runner.is_hybrid_swa:
            return
        if not envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
            return
        if not envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            return

        max_num_tokens = model_runner.server_args.chunked_prefill_size
        if max_num_tokens is None or max_num_tokens <= 0:
            max_num_tokens = 8192

        token_counts = self.prewarm_mhc_token_count_buckets(
            max_num_tokens, model_runner.device
        )
        model_runner.tp_group.barrier()

        logger.info(
            "DeepSeek V4 MHC prewarm completed for representative token-count shapes: %s",
            token_counts,
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def determine_num_fused_shared_experts(self):
        self.num_fused_shared_experts = 0
        if get_global_server_args().disable_shared_experts_fusion:
            return

        get_global_server_args().disable_shared_experts_fusion = True
        log_info_on_rank0(
            logger,
            "DeepSeek V4 requires different clamping for shared and routed experts. "
            "Shared experts fusion optimization is disabled.",
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.nsa_enable_prefill_cp:
            if can_nsa_cp_split(len(input_ids), self.cp_size, True, forward_batch):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                )
                if is_nsa_prefill_cp_round_robin_split():
                    metadata = forward_batch.attn_backend.forward_metadata
                    core_meta = metadata.core_attn_metadata
                    core_meta.apply_cp_reindex()
                    core_meta.init_flashmla_related()
                    if metadata.indexer_metadata is not None:
                        metadata.indexer_metadata = (
                            forward_batch.attn_backend.init_forward_metadata_indexer(
                                core_meta
                            )
                        )

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model.forward(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        if not self.pp_group.is_last_rank:
            return hidden_states

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        hidden_states, pre_hc_head = hidden_states
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
            hidden_states_before_norm=pre_hc_head,
        )

    def _setup_fp8_wo_a_scales(self, is_nextn: bool) -> None:
        from deep_gemm import transform_sf_into_required_layout

        if is_nextn:
            layers = [self.model.decoder]
        else:
            layers = [
                self.model.layers[layer_id]
                for layer_id in range(self.model.start_layer, self.model.end_layer)
            ]
        for layer in layers:
            attn = layer.self_attn
            G = attn.n_local_groups
            R = attn.o_lora_rank
            D = attn.wo_a.weight.shape[1]

            raw_scale = attn.wo_a.weight_scale_inv.data.view(G, R // 128, D // 128)
            attn.wo_a.weight_scale_inv.data = transform_sf_into_required_layout(
                raw_scale,
                mn=R,
                k=D,
                recipe=(1, 128, 128),
                num_groups=G,
                is_sfa=False,
            )

    def post_load_weights(self, is_nextn=False, weight_names=None):
        if _FP8_WO_A_GEMM:
            self._setup_fp8_wo_a_scales(is_nextn)

        if is_nextn:
            return
        for layer_id in range(self.model.start_layer, self.model.end_layer):
            layer = self.model.layers[layer_id]
            self_attn = layer.self_attn
            if self_attn.compress_ratio != 0 and not self_attn.compressor.ape_converted:
                self_attn.compressor.apply_ape_hotfix()
            if (
                self_attn.compress_ratio == 4
                and not self_attn.indexer.compressor.ape_converted
            ):
                self_attn.indexer.compressor.apply_ape_hotfix()

    @staticmethod
    def remap_weight_name_to_dpsk_hf_format(
        name: str, is_nextn: bool = False, num_hidden_layers: Optional[int] = None
    ) -> str:
        if name == "embed.weight":
            return "model.embed_tokens.weight"
        if name == "head.weight":
            return "lm_head.weight"
        if name == "norm.weight":
            return "model.norm.weight"
        if name.startswith("hc_head_"):
            return "model." + name

        if is_nextn and name.startswith("mtp."):
            parts = name.split(".", 2)
            if len(parts) >= 3:
                rest = parts[2]
                nextn_spec_prefixes = [
                    "e_proj",
                    "h_proj",
                    "emb",
                    "enorm",
                    "hnorm",
                    "norm",
                    "head",
                    "hc_head",
                ]
                is_nextn_spec = any(rest.startswith(p) for p in nextn_spec_prefixes)
                if is_nextn_spec:
                    if rest.startswith("emb.tok_emb"):
                        rest = rest.replace("emb.tok_emb", "embed_tokens")
                    elif rest == "norm.weight":
                        rest = "shared_head.norm.weight"
                    elif rest.startswith("head."):
                        rest = "shared_head.head.weight"
                    elif rest == "e_proj.scale":
                        rest = "e_proj.weight_scale_inv"
                    elif rest == "h_proj.scale":
                        rest = "h_proj.weight_scale_inv"
                name = f"model.layers.{num_hidden_layers}." + rest

        if name.startswith("layers."):
            name = "model." + name
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

        if "self_attn" in name:
            name = name.replace(".scale", ".weight_scale_inv")

        name = name.replace(".gate.tid2eid", ".topk.tid2eid")
        name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
        name = name.replace(".w1.", ".gate_proj.")
        name = name.replace(".w2.", ".down_proj.")
        name = name.replace(".w3.", ".up_proj.")
        if "mlp" in name:
            name = name.replace(".scale", ".weight_scale_inv")

        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        if not envs.SGLANG_OPT_FP8_WO_A_GEMM.get():
            weights = list(weights)
            exists_wo_a_scale = any(n.endswith(".wo_a.scale") for n, t in weights)
            if exists_wo_a_scale:
                logger.info("Execute dequant fp8 wo_a")
                weights = _dequant_fp8_wo_a(weights)
            else:
                logger.info("Skip dequant fp8 wo_a")

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        if self.quant_config and self.quant_config.get_name() == "w4afp8":
            expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
                num_experts=self.config.n_routed_experts
            )

        cache_compressor_weight = {}
        COMPRESSOR_PART = ".compressor.w"

        fuse_wqa_wkv = envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        cache_wqkv_a_weight: dict[str, dict[str, torch.Tensor]] = {}

        def auto_weight_loader(module):
            return getattr(module, "weight_loader", default_weight_loader)

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names_out_of_layer = [
                "shared_head.norm",
                "shared_head.head",
                "embed_tokens",
                ".e_proj",
                "h_proj",
                "enorm",
                "hnorm",
                "hc_head_base",
                "hc_head_fn",
                "hc_head_scale",
            ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            weight_names = []
            for name, loaded_weight in weights:
                try:
                    use_async_loading = should_async_load(loaded_weight)

                    name = self.remap_weight_name_to_dpsk_hf_format(
                        name,
                        is_nextn=is_nextn,
                        num_hidden_layers=self.config.num_hidden_layers,
                    )

                    layer_id = get_layer_id(name)
                    if (
                        layer_id is not None
                        and hasattr(self.model, "start_layer")
                        and (
                            layer_id < self.model.start_layer
                            or layer_id >= self.model.end_layer
                        )
                    ):
                        continue
                    if (
                        self.num_fused_shared_experts > 0
                        and "mlp.shared_experts" in name
                    ):
                        name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts}",
                        )

                    weight_names.append(name)

                    if not is_nextn:
                        if hasattr(self.config, "num_nextn_predict_layers"):
                            num_nextn_layers = self.config.num_nextn_predict_layers
                            if num_nextn_layers > 0 and name.startswith("model.layers"):
                                name_list = name.split(".")
                                if (
                                    len(name_list) >= 3
                                    and int(name_list[2])
                                    >= self.config.num_hidden_layers
                                ):
                                    continue

                            if name.startswith("mtp"):
                                continue
                    else:
                        if "shared_head.head" in name or "embed_tokens" in name:
                            continue

                        if not name.startswith(nextn_layer_prefix):
                            continue

                        in_decoder = True
                        for weight_name in nextn_spec_weight_names_out_of_layer:
                            if weight_name in name:
                                in_decoder = False
                                name = name.replace(nextn_layer_prefix, "model")
                                break

                        if in_decoder:
                            name = name.replace(nextn_layer_prefix, "model.decoder")

                    if "rotary_emb.inv_freq" in name:
                        continue
                    for param_name, weight_name, shard_id in stacked_params_mapping:
                        if weight_name not in name:
                            continue
                        if _is_npu:
                            name = name.replace("weight_packed", "weight")
                        if ("mlp.experts." in name) and name not in params_dict:
                            continue
                        name = name.replace(weight_name, param_name)
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if name not in params_dict and name.startswith("mtp"):
                            break
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(param, loaded_weight, shard_id),
                        )
                        loaded_params.add(name)
                        break
                    else:
                        for mapping in expert_params_mapping:
                            param_name, weight_name, expert_id, shard_id = mapping
                            if weight_name not in name:
                                continue
                            if _is_npu:
                                name = name.replace("weight_packed", "weight")
                            name = name.replace(weight_name, param_name)
                            if name not in params_dict:
                                continue
                            param = params_dict[name]
                            weight_loader = param.weight_loader
                            maybe_executor_submit(
                                executor=executor,
                                futures=futures,
                                use_async=use_async_loading,
                                func=weight_loader,
                                func_args=(
                                    param,
                                    loaded_weight,
                                    name,
                                ),
                                func_kwargs={
                                    "shard_id": shard_id,
                                    "expert_id": expert_id,
                                },
                            )
                            loaded_params.add(name)
                            break
                        else:
                            if name.endswith(".bias") and name not in params_dict:
                                continue
                            if (
                                ".embed_tokens." in name
                                and not self.pp_group.is_first_rank
                            ):
                                continue
                            if (
                                name == "model.norm.weight"
                                and not self.pp_group.is_last_rank
                            ):
                                continue
                            if (
                                name.startswith("model.hc_head_")
                                or name == "lm_head.weight"
                            ) and not self.pp_group.is_last_rank:
                                continue
                            elif COMPRESSOR_PART in name:
                                is_kv = name.endswith(".wkv.weight")
                                is_wgate = name.endswith(".wgate.weight")
                                assert is_kv != is_wgate
                                key = name.rsplit(".", 2)[0]
                                assert key.endswith(".compressor")
                                if key not in cache_compressor_weight:
                                    cache_compressor_weight[key] = (
                                        is_kv,
                                        loaded_weight,
                                    )
                                else:
                                    assert key in cache_compressor_weight
                                    cached_is_kv, cached_weight = (
                                        cache_compressor_weight[key]
                                    )
                                    assert cached_is_kv != is_kv
                                    kv = loaded_weight if is_kv else cached_weight
                                    wgate = loaded_weight if is_wgate else cached_weight
                                    fused_weight = torch.cat([kv, wgate], dim=0)
                                    param_name = key + ".wkv_gate.weight"
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_compressor_weight.pop(key)
                            elif fuse_wqa_wkv and (
                                name.endswith(".wq_a.weight")
                                or name.endswith(".wq_a.weight_scale_inv")
                                or name.endswith(".wkv.weight")
                                or name.endswith(".wkv.weight_scale_inv")
                            ):
                                is_q = ".wq_a." in name
                                param_name = name.replace(
                                    ".wq_a." if is_q else ".wkv.", ".wqkv_a."
                                )
                                bucket = cache_wqkv_a_weight.setdefault(param_name, {})
                                shard_key = "q" if is_q else "kv"
                                assert (
                                    shard_key not in bucket
                                ), f"duplicate shard {shard_key} for {param_name}"
                                bucket[shard_key] = loaded_weight
                                if len(bucket) == 2:
                                    fused_weight = torch.cat(
                                        [bucket["q"], bucket["kv"]], dim=0
                                    )
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_wqkv_a_weight.pop(param_name)
                            else:
                                if (
                                    "k_scale" in name or "v_scale" in name
                                ) and name not in params_dict:
                                    for scale in ["k_scale", "v_scale"]:
                                        if scale in name:
                                            name = name.replace(
                                                f"{scale[0]}_proj", "attn_mqa"
                                            )
                                            break
                                if name not in params_dict:
                                    if not name.startswith("mtp"):
                                        logger.warning(
                                            f"{name} not found in params_dict."
                                        )
                                    continue
                                param = params_dict[name]

                                weight_loader = auto_weight_loader(param)
                                maybe_executor_submit(
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, loaded_weight),
                                )
                                loaded_params.add(name)
                except Exception as e:
                    e.add_note(f"{name=} {loaded_weight.shape=}")
                    raise

            for future in concurrent.futures.as_completed(futures):
                future.result()

        assert len(cache_compressor_weight) == 0
        assert len(cache_wqkv_a_weight) == 0, cache_wqkv_a_weight.keys()
        unloaded_params = params_dict.keys() - loaded_params

        skipped_checking_patterns = ["attn_mqa.k_scale", "attn_mqa.v_scale"]
        if not self.pp_group.is_first_rank:
            skipped_checking_patterns.append("embed_tokens")
        if not self.pp_group.is_last_rank:
            skipped_checking_patterns.append("model.norm.")
            skipped_checking_patterns.extend(["lm_head", "hc_head_"])
        if is_nextn:
            skipped_checking_patterns.extend(["lm_head", "embed_tokens"])
        unloaded_params = {
            p
            for p in unloaded_params
            if all(
                skipped_checking_pattern not in p
                for skipped_checking_pattern in skipped_checking_patterns
            )
        }
        if unloaded_params:
            logger.warning(
                f"Some weights are not initialized from checkpoints: {unloaded_params}"
            )

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )


EntryClass = [DeepseekV4ForCausalLM]


def _dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from einops import rearrange

    assert (
        weight.dtype == torch.float8_e4m3fn
    ), f"expected fp8_e4m3fn, got {weight.dtype}"
    assert scale.dtype in (
        torch.float8_e8m0fnu,
        torch.float32,
    ), f"expected fp8_e8m0fnu or float32, got {scale.dtype}"

    weight_f32 = rearrange(
        weight.float(), "(sn bn) (sk bk) -> sn bn sk bk", bn=128, bk=128
    )
    result = rearrange(
        weight_f32 * scale.float()[:, None, :, None], "sn bn sk bk -> (sn bn) (sk bk)"
    )

    return result.to(torch.bfloat16)


def _dequant_fp8_wo_a(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterable[Tuple[str, torch.Tensor]]:
    weights_dict = dict(weights)

    for name in list(weights_dict.keys()):
        if name not in weights_dict:
            continue
        if not name.endswith(".wo_a.weight"):
            continue
        scale_name = name.replace(".wo_a.weight", ".wo_a.scale")
        assert scale_name in weights_dict
        weight = weights_dict.pop(name)
        scale = weights_dict.pop(scale_name)
        yield name, _dequant_fp8(weight, scale)

    yield from weights_dict.items()
