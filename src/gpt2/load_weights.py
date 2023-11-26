"""
Load GPT-2 weights
"""

import re

import numpy as np
from safetensors import safe_open
import tqdm

LAYER_NAMES = {
    "attn.c_proj.bias": "attention.proj.bias",
    "attn.c_proj.weight": "attention.proj.weights",
    # "attn.c_attn.bias": "attention.query.bias",
    # "attn.c_attn.weight": "attention.query.weight",
    "ln_1.bias": "ln1.bias",
    "ln_1.weight": "ln1.scale",
    "ln_2.bias": "ln2.bias",
    "ln_2.weight": "ln2.scale",
    "mlp.c_fc.bias": "ff.linear1.bias",
    "mlp.c_fc.weight": "ff.linear1.weights",
    "mlp.c_proj.bias": "ff.linear2.bias",
    "mlp.c_proj.weight": "ff.linear2.weights",
    "ln_f.bias": "ln_final.bias",
    "ln_f.weight": "ln_final.scale",
    # "wpe.weight": "position_embeddings",
    # "wte.weight": "embeddings",
}

with safe_open("model.safetensors", framework="numpy") as model:
    tensors = {}

    for tensor_name in tqdm.tqdm(model.keys()):
        tensor = model.get_tensor(tensor_name)

        # Force dimensions to 3D
        while len(tensor.shape) > 3:
            tensor = np.squeeze(tensor, axis=0)
        while len(tensor.shape) < 3:
            tensor = np.expand_dims(tensor, axis=0)

        # Parse tensor name
        match = re.match(r"h\.(\d+)\.(.+)", tensor_name)

        if match:
            block = int(match.group(1))
            layer = LAYER_NAMES.get(match.group(2), match.group(2))

            if layer == "attn.bias":
                continue
            elif layer == "attn.c_attn.bias":
                # Split bias into query, key, value
                query, key, value = np.split(tensor, 3, axis=-1)

                tensors[f"block{block}.attention.query.bias"] = query
                tensors[f"block{block}.attention.key.bias"] = key
                tensors[f"block{block}.attention.value.bias"] = value
            elif layer == "attn.c_attn.weight":
                # Split weight into query, key, value
                query, key, value = np.split(tensor, 3, axis=-1)

                tensors[f"block{block}.attention.query.weights"] = query
                tensors[f"block{block}.attention.key.weights"] = key
                tensors[f"block{block}.attention.value.weights"] = value
            else:
                tensors[f"block{block}.{layer}"] = tensor

        else:
            block = None
            layer = LAYER_NAMES.get(tensor_name, tensor_name)

            if layer == "wpe.weight":
                chunk1 = tensor[:, :tensor.shape[1] // 2, :]
                chunk2 = tensor[:, tensor.shape[1] // 2:, :]

                tensors["position_embeddings.chunk1"] = chunk1
                tensors["position_embeddings.chunk2"] = chunk2
            elif layer == "wte.weight":
                chunk1 = tensor[:, :tensor.shape[1] // 2, :]
                chunk2 = tensor[:, tensor.shape[1] // 2:, :]

                tensors["embeddings.chunk1"] = chunk1
                tensors["embeddings.chunk2"] = chunk2
            else:
                tensors[layer] = tensor

    for tensor_name, tensor in tensors.items():
        print(tensor_name, tensor.shape, tensor[0, 0, :5])
      
        header = np.array(tensor.shape, dtype=np.uint32)

        with open(f"../../public/gpt2_weights/{tensor_name}.bin", "wb") as f:
            f.write(header.tobytes())
            f.write(tensor.astype(np.float32).tobytes())