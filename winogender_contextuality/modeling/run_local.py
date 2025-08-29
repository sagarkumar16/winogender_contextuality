import torch
from torch import bfloat16
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel, AutoTokenizer, AutoConfig
from transformers.quantizers import Mxfp4Config
import huggingface_hub
from winogender_contextuality.config import *


# Check CUDA
logger.info(f'torch available: {torch.cuda.is_available()}', flush=True)
logger.info(torch.version.cuda)
for i in range(torch.cuda.device_count()):
    logger.info(torch.cuda.get_device_properties(i))

def load_model(model_name: str,
               api_key: str,
               quantized: bool,
               model_path: str) -> PreTrainedModel:
    """
    Empties cache and loads a model locally then caches it in a folder called "cache" in the model_path directory
    :param model_name: huggingface model name
    :param api_key: huggingface api key
    :param quantized: whether to quantize the model
    :param model_path: directory where model is cached
    :return: a huggingface model
    """

    cache_dir = Path(model_path) / "cache"   # <-- fix: ensure correct Path usage
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Connect with Hugging Face
    huggingface_hub.login(api_key)
    logger.info("Connected with Hugging Face", flush=True)

    if not quantized:
        logger.info("Loading full-precision model", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=api_key,
            cache_dir=cache_dir,
            device_map="auto",
        )
        model.config.use_cache = False
        logger.info(f"Model cached in {cache_dir}", flush=True)
        return model

    # -------- Quantized path --------
    # Choose quantization backend:
    # - gpt-oss -> MXFP4
    # - others  -> bitsandbytes 4-bit
    use_mxfp4 = "gpt-oss" in model_name.lower()

    if use_mxfp4:
        logger.info("Loading quantized model with MXFP4 (gpt-oss detected)", flush=True)

        # IMPORTANT: do NOT pass BitsAndBytesConfig at the same time.
        qcfg = Mxfp4Config()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=api_key,
            cache_dir=cache_dir,
            device_map="auto",
            quantization_config=qcfg,   # only MXFP4
        )
        model.config.use_cache = False
        logger.info(f"Model cached in {cache_dir}", flush=True)
        return model

    # Default: bitsandbytes 4-bit
    logger.info("Loading quantized model with bitsandbytes 4-bit", flush=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # H100/H200-friendly
    )

    # Ensure we don't inherit a repo-side quantization_config that conflicts
    try:
        cfg = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, token=api_key)
        if hasattr(cfg, "quantization_config"):
            setattr(cfg, "quantization_config", None)
    except Exception:
        pass  # if config fetch fails, proceed anyway

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=api_key,
        cache_dir=cache_dir,
        device_map="auto",
        quantization_config=bnb_config,   # only bnb
    )
    model.config.use_cache = False

    logger.info(f"Model cached in {cache_dir}", flush=True)
    return model