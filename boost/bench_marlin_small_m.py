"""How close is Marlin W4A16 (zero-point) to the memory-bandwidth floor at decode M?

Times the kernel on this checkpoint's four GEMM shapes at M = 1, 5, 8, 16 and reports
achieved GB/s against the bytes it has to read (packed int4 + bf16 scales + packed zp).
Decides whether a hand-written small-M GEMV is worth writing.
"""
import torch, time
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, ensure_model_parallel_initialized
_c = set_current_vllm_config(VllmConfig()); _c.__enter__()
init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:29519", local_rank=0)
ensure_model_parallel_initialized(1, 1)
from vllm.model_executor.kernels.linear.mixed_precision import MPLinearLayerConfig
from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.scalar_type import scalar_types
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32

dev = "cuda"; G = 128
def layer(K, N, act):
    q = torch.randint(-8, 8, (N, K), dtype=torch.int8)
    zp = torch.randint(-8, 8, (N, K // G), dtype=torch.int8)
    s = (torch.rand(N, K // G) * 0.01 + 0.001).to(torch.bfloat16)
    l = torch.nn.Module(); wl = lambda *a, **k: None
    l.weight_packed = PackedvLLMParameter(data=pack_to_int32(q, 4, packed_dim=1).to(dev), input_dim=1, output_dim=0, packed_dim=1, packed_factor=8, weight_loader=wl)
    l.weight_scale = GroupQuantScaleParameter(data=s.to(dev), output_dim=0, input_dim=1, weight_loader=wl)
    l.weight_zero_point = PackedvLLMParameter(data=pack_to_int32(zp, 4, packed_dim=0).to(dev), input_dim=1, output_dim=0, packed_dim=0, packed_factor=8, weight_loader=wl)
    cfg = MPLinearLayerConfig(full_weight_shape=(K, N), partition_weight_shape=(K, N), weight_type=scalar_types.uint4,
                              act_type=act, group_size=G, zero_points=True, has_g_idx=False)
    k = MarlinLinearKernel(cfg, w_q_param_name="weight_packed", w_s_param_name="weight_scale", w_zp_param_name="weight_zero_point", w_gidx_param_name="weight_g_idx")
    k.process_weights_after_loading(l)
    return l, k

def bench(K, N, M, act, iters=200):
    l, k = layer(K, N, act)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    for _ in range(20): k.apply_weights(l, x)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters): k.apply_weights(l, x)
    en.record(); torch.cuda.synchronize()
    us = st.elapsed_time(en) * 1000 / iters
    nbytes = N * K // 2 + N * (K // G) * 2 + N * (K // G) // 2 + M * K * 2 + M * N * 2
    return us, nbytes / us / 1e3  # GB/s

print(f"{'shape':>14} {'M':>3} {'act':>5} {'us':>8} {'GB/s':>7}")
for K, N in [(5120, 17408), (17408, 5120), (5120, 12288), (5120, 6144)]:
    for M in (1, 5, 8, 16):
        for act in (torch.bfloat16, torch.int8):
            us, gbs = bench(K, N, M, act)
            print(f"{K:>6}x{N:<7} {M:>3} {'int8' if act is torch.int8 else 'bf16':>5} {us:8.1f} {gbs:7.0f}", flush=True)
# the whole body per decode step: 48 GDN layers + 16 attention layers, at M=1
print("theoretical floor for the int4 body (12.3 GB at 936 GB/s):", f"{12.3e9/936e9*1e3:.1f} ms/step")
