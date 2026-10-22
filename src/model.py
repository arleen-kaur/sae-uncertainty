import yaml
import torch
import operator
from functools import reduce
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


# Path to the YAML config file containing model-specific settings
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "models.yaml" 


# Load yamal config for the given model. 
def load_model_config(model_name: str) -> dict:
    with open(CONFIG_PATH) as f:
        configs = yaml.safe_load(f)
    if model_name not in configs:
        raise ValueError(
            f"Model '{model_name}' not found in {CONFIG_PATH}. "
            f"Available: {list(configs.keys())}"
        )
    return configs[model_name]


# get the list of layers from model object
def get_layers(model, config: dict):
    """Get the model's transformer layers using the configured accessor path."""
    parts = config["layer_accessor"].split(".")
    obj = model
    for part in parts:
        obj = getattr(obj, part)
    return obj


def load_model(model_name: str, device: str = "cuda"):
    """Load model and tokenizer. Returns (model, tokenizer, config)."""
    config = load_model_config(model_name)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    
    #get dtype from config, default to float16 if not specified
    dtype = dtype_map.get(config.get("dtype", "float16"), torch.float16)

    print(f"Loading model and tokenizer for {model_name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()

    layers = get_layers(model, config)
    num_layers = len(layers)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9

    print(f"Model loaded successfully")
    print(f"  - Device: {model.device}")
    print(f"  - dtype: {model.dtype}")
    print(f"  - Parameters: {n_params:.2f}B")
    print(f"  - Layers: {num_layers}")

    return model, tokenizer, config


DEFAULT_LETTERS = ['A', 'B', 'C', 'D']


def get_letter_token_ids(tokenizer, letters=None) -> dict:
    """Map answer letters to their token IDs.

    `letters` defaults to A/B/C/D (original MCQ behaviour). Pass e.g. ['A', 'B']
    for binary eval sets (see `src.utils.get_eval_letters`). Entropy and
    prediction code iterate over the returned dict, so any number of options works.
    """
    if letters is None:
        letters = DEFAULT_LETTERS
    letter_token_ids = {}
    for letter in letters:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if ids:
            letter_token_ids[letter] = ids[-1]
    return letter_token_ids


def get_default_sae_layers(config: dict) -> list:
    return config.get("default_sae_layers", [])
