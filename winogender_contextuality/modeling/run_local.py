import torch
from torch import bfloat16
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel, AutoTokenizer, AutoConfig
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

    cache_dir = Path(model_path / 'cache')
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Connect with Huggingface
    huggingface_hub.login(api_key)
    logger.info('Connected with Huggingface', flush=True)

    if not quantized:
        # full model
        logger.info('Loading full model', flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=api_key, cache_dir=cache_dir, device_map="auto")
        model.config.use_cache = False
    else:
        # quantized version
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # note: use torch.bfloat16
        )

        cfg = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        # remove model-advertised quantization (e.g., MXFP4)
        if hasattr(cfg, "quantization_config"):
            setattr(cfg, "quantization_config", None)

        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            cache_dir=cache_dir,
        )
        model.config.use_cache = False

    logger.info(f'Model cached in {cache_dir}', flush=True)

    return model