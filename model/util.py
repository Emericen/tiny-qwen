import os
import json
from pathlib import Path
from typing import Optional, Union, Dict
import torch
from huggingface_hub import snapshot_download, logging
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


from .vision import VisionConfig

logging.set_verbosity_error()


def _rename_dict_keys(original_dict: dict, key_mapping: dict) -> dict:
    """
    Renames keys in a dictionary according to a provided mapping.

    Args:
        original_dict (dict): The original dictionary whose keys need to be renamed.
        key_mapping (dict): A mapping from old key names to new key names (old_key_name: new_key_name).

    Returns:
        dict: A new dictionary with keys renamed according to the mapping.
              Keys not present in the mapping remain unchanged.
    """
    new_dict = {}
    for key, value in original_dict.items():
        new_key_name = key_mapping[key] if key in key_mapping else key
        new_dict[new_key_name] = value
    return new_dict


def _convert_llm_config(hf_config: dict):
    llm_config_key_name_mapping = {
        "hidden_size": "n_embed",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "n_kv_heads",
        "num_hidden_layers": "n_layer",
        "intermediate_size": "n_mlp",
        "rms_norm_eps": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "rope_theta": "rope_theta",
        "tie_word_embeddings": "tie_word_embeddings",
        "head_dim": "head_dim",
    }
    return _rename_dict_keys(
        original_dict=hf_config,
        key_mapping=llm_config_key_name_mapping,
    )


def _convert_vision_config(hf_config: dict):
    vision_config = hf_config["vision_config"]

    # Handle different naming conventions between Qwen2-VL and Qwen2.5-VL
    if "embed_dim" in vision_config:
        # Qwen2-VL format
        vision_config_key_name_mapping = {
            "depth": "n_layer",
            "embed_dim": "n_embed",
            "num_heads": "n_heads",
            "in_chans": "in_channels",
            "hidden_size": "output_n_embed",
            "spatial_patch_size": "spatial_patch_size",
            "temporal_patch_size": "temporal_patch_size",
            "spatial_merge_size": "spatial_merge_size",
        }
    else:
        # Qwen2.5-VL format
        vision_config_key_name_mapping = {
            "depth": "n_layer",
            "hidden_size": "n_embed",
            "num_heads": "n_heads",
            "out_hidden_size": "output_n_embed",
            "in_chans": "in_channels",  # Qwen2.5-VL also uses "in_chans"
            "patch_size": "spatial_patch_size",
            "temporal_patch_size": "temporal_patch_size",
            "spatial_merge_size": "spatial_merge_size",
            "intermediate_size": "intermediate_size",  # For gated MLP
            "hidden_act": "hidden_act",  # Activation function
        }

    return _rename_dict_keys(
        original_dict=vision_config,
        key_mapping=vision_config_key_name_mapping,
    )


def _filter_dict_by_dataclass(params: dict, dataclass_type) -> dict:
    return {k: v for k, v in params.items() if k in dataclass_type.__annotations__}


def load_pretrained_model(
    model_cls,
    repo_id: Union[str, Path],
    device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = "auto",
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    force_download: bool = False,
    **kwargs,
):
    """
    Load a pretrained model using the same logic as the mixin, but as a standalone function.

    Args:
        model_cls: The model class (Qwen2VL, Qwen2, etc.)
        repo_id: HuggingFace repo ID or local path
        device_map: Device mapping for multi-GPU

    Returns:
        Loaded model instance
    """
    # Determine model path
    if os.path.isdir(repo_id):
        model_path = Path(repo_id)
    else:
        model_path = Path(
            snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                force_download=force_download,
            )
        )

    # Load config
    with open(model_path / "config.json", "r") as f:
        config_data = json.load(f)

    # Get the appropriate config class from the model
    config_cls = model_cls.get_config_class()

    llm_config = _convert_llm_config(config_data)
    llm_config = _filter_dict_by_dataclass(llm_config, config_cls)

    model_config = config_cls(**llm_config)
    if "vision_config" in config_data:
        vision_config = _convert_vision_config(config_data)
        vision_config = _filter_dict_by_dataclass(vision_config, VisionConfig)
        model_config.vision_config = VisionConfig(**vision_config)

    # Different initialization for vision vs text-only models
    if model_cls.__name__ == "Qwen2VL":
        # Vision models - don't use init_empty_weights due to issues
        model = model_cls(model_config)
        model = load_checkpoint_and_dispatch(
            model,
            model_path,
            device_map=device_map,
            dtype=torch.bfloat16,
            no_split_module_classes=[
                "Block",
                "Qwen2VLVisionBlock",
                "Qwen2VLVisionEncoder",
                "PatchEmbed",
                "PatchMerger",
                "VisionMlp",
                "VisionAttention",
                "VisionRotaryEmbedding",
            ],
        )
    elif model_cls.__name__ == "Qwen3MoE":
        # MoE models - use init_empty_weights but with special handling
        with init_empty_weights():
            model = model_cls(model_config)
        try:
            model = load_checkpoint_and_dispatch(
                model,
                model_path,
                device_map=device_map,
                dtype=torch.bfloat16,
                no_split_module_classes=["Block", "Qwen3Block"],
            )
        except NotImplementedError as e:
            if "Cannot copy out of meta tensor" in str(e):
                print("Meta tensor issue detected, trying alternative loading...")
                # Fallback: create model normally and let it OOM if needed
                model = model_cls(model_config)
                model = load_checkpoint_and_dispatch(
                    model,
                    model_path,
                    device_map=device_map,
                    dtype=torch.bfloat16,
                    no_split_module_classes=["Block", "Qwen3Block"],
                )
            else:
                raise
    else:
        # Dense text-only models - can use init_empty_weights
        with init_empty_weights():
            model = model_cls(model_config)
        model = load_checkpoint_and_dispatch(
            model,
            model_path,
            device_map=device_map,
            dtype=torch.bfloat16,
            no_split_module_classes=["Block", "Qwen3Block"],
        )

    return model
