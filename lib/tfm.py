from huggingface_hub import hf_hub_download
from torch import nn
import hashlib
from pathlib import Path

import lib
import lib.limix.utils.loading


def load_tfm(
    tfm_name: str,
    tfm_config: dict,
) -> nn.Module:
    if tfm_name == "LimiX":
        model_path = Path('./checkpoints/LimiX-16M.ckpt')
        if not model_path.is_file():
            model_path = Path(hf_hub_download(
                repo_id="stableai-org/LimiX-16M",
                filename="LimiX-16M.ckpt",
                revision="da5f3072bf3633c70d957c02518c30d461007764",
                local_dir="./checkpoints",
                cache_dir="./checkpoints",
            ))
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != "ee6d6ae865821ca790a40a0199d09a662f4509da5e4c26535809432e599e7a52":
            raise ValueError(f'LimiX checkpoint hash mismatch: {model_path}')
        return lib.limix.utils.loading.load_model(model_path=model_path)
    else:
        raise ValueError(f"{tfm_name} is not found")
