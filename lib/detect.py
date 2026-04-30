import sys
import argparse
import json
from vllm.transformers_utils.config import get_config

def get_tc(model):
    config = get_config(model=model, trust_remote_code=True)
    return getattr(config, 'text_config', config)

def detect_arch(model):
    tc = get_tc(model)
    # Check for common MoE expert count attributes
    E = getattr(tc, 'num_local_experts', 0) or getattr(tc, 'n_routed_experts', 0)
    return 'moe' if int(E) > 0 else 'dense'

def detect_shapes(model, tp):
    tc = get_tc(model)
    hidden_size = getattr(tc, 'hidden_size', None)
    if not hidden_size:
        return []

    num_heads = getattr(tc, 'num_attention_heads', 16)
    num_kv_heads = getattr(tc, 'num_key_value_heads', num_heads)
    head_dim = getattr(tc, 'head_dim', hidden_size // num_heads)
    shared_expert_size = getattr(tc, 'shared_expert_intermediate_size', None)
    intermediate_size = getattr(tc, 'intermediate_size', None)
    moe_intermediate_size = getattr(tc, 'moe_intermediate_size', None)

    shapes = set()
    
    # QKV projection
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_out = (q_dim + 2 * kv_dim) // tp
    shapes.add((qkv_out, hidden_size))

    # Attention output
    shapes.add((hidden_size, q_dim // tp))

    # Linear attention projections (Mamba-style)
    lin_key_heads = getattr(tc, 'linear_num_key_heads', None)
    lin_val_heads = getattr(tc, 'linear_num_value_heads', None)
    lin_key_dim = getattr(tc, 'linear_key_head_dim', None)
    lin_val_dim = getattr(tc, 'linear_value_head_dim', None)
    if all(v is not None for v in [lin_key_heads, lin_val_heads, lin_key_dim, lin_val_dim]):
        lin_out = (lin_key_heads * lin_key_dim + lin_val_heads * lin_val_dim +
                   lin_key_heads * lin_key_dim)
        shapes.add((lin_out // tp, hidden_size))

    # Shared expert
    if shared_expert_size:
        shapes.add((shared_expert_size, hidden_size))
        shapes.add((hidden_size, shared_expert_size // tp))

    # Dense FFN
    if intermediate_size:
        shapes.add((intermediate_size // tp, hidden_size))
        shapes.add((hidden_size, intermediate_size // tp))

    # MoE experts
    if moe_intermediate_size:
        shapes.add((moe_intermediate_size, hidden_size))
        shapes.add((hidden_size, moe_intermediate_size // tp))

    return sorted(list(shapes))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--mode", choices=["arch", "shapes", "all"], default="all")
    args = parser.parse_args()

    result = {}
    if args.mode in ["arch", "all"]:
        result["arch"] = detect_arch(args.model)
    
    if args.mode in ["shapes", "all"]:
        shapes = detect_shapes(args.model, args.tp)
        result["shapes"] = [f"{n},{k}" for n, k in shapes]

    print(json.dumps(result))

if __name__ == "__main__":
    main()
