from transformers import AutoTokenizer
import torch
from loguru import logger
from winogender_contextuality.modeling.run_local import load_model
from winogender_contextuality.utils import flush
from winogender_contextuality.config import *

"""
Base class for usage in contextuality measurement or metaprompting.

- Sagar Kumar, 2025
"""


class ModelProbs:

    def __init__(self,
                 mode: str,
                 model_name: str,
                 key: str,
                 model_path: str = None,
                 quantized: bool = True
                 ):

        self.mode = mode # self.mode = True maps to API call instead of local
        self.model_name = model_name
        self.key = key
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=key)


        # Local Run Attributes
        self.quantized = quantized
        self.model_path = model_path
        self.model = None
        self.gpu = GPU_INDEX


    # TODO: if API, use pipeline
    def load_model(self) -> None:

        """
        Loads a huggingface pretrained model locally, sets that as self.model, and caches it in the model_path directory
        :return: None
        """

        if self.mode == 'api':
            raise NotImplementedError

        elif self.mode == 'gpu':
            if self.model_path is None:
                raise AttributeError("Must specify a model path for local runs")
            else:
                flush()
                self.model = load_model(self.model_name, self.key, self.quantized, self.model_path)
                logger.info(f"Loaded model {self.model_name} to {self.model_path}")

        else:
            raise AttributeError("Mode must be either 'api' or 'gpu'")

    def get_raw_logits(self,
                       prompt: str):

        """
        Obtains raw logits (over the entire vocabulary) for the next token
        :param prompt: formatted prompt string
        :return: logits over the entire vocabulary for the next token
        """

        if self.mode == 'api':
            raise NotImplementedError

        else:
            inputs = (self.tokenizer.apply_chat_template(prompt, return_tensors="pt", continue_final_message=True)
                      .to("cuda:0"))
            with torch.no_grad():
                outputs = self.model(inputs)

            logits = outputs.logits
            next_token_logits = logits[0, -1]

        return next_token_logits

    def get_token_ids(self,
                      options: list[str]) -> list[list[int]]:
        """
        Outputs token IDs
        :param options: list of tokens
        :return: list of token id lists
        """

        assert all([s[0] == " " for s in options]), "Tokens must begin with a space. "

        token_ids = [self.tokenizer.encode(opt, add_special_tokens=False) for opt in options]

        for n, id in enumerate(token_ids):
            if len(id) > 1:
                logger.info(f"{options[n]} decomposes into two tokens.")

        return token_ids

    # TODO: prompting generation method
    #  - this will have to do all the first_id nonsense to make sure we are looking at the correct probs
    def get_completion(self,
                       prompt: str,
                       **kwargs):

        """
        Get text generated output based on prompt. kwargs specify changes to default params in model.generate()
        :param prompt: formatted prompt string
        :param kwargs: model.generation() parameters
        :return: generated output object
        """

        default_args = {
            'output_scores': True,
            'return_dict_in_generate': True,
            'output_hidden_states': True,
            'do_sample': True,
            'pad_token_id': self.tokenizer.eos_token_id,
            'max_new_tokens': 6,
            'temperature': 0.5,
            'top_k': 2
        }

        # If kwargs were provided, update the defaults -- this is where MODEL_PARAMS.params from config go
        generation_args = {**default_args, **kwargs}

        if self.mode == 'api':
            raise NotImplementedError

        else:
            inputs = (self.tokenizer.apply_chat_template(prompt, return_tensors="pt", continue_final_message=True)
                      .to(self.gpu))

            outputs = self.model.generate(inputs, **generation_args)

        return outputs

    def get_completed_logits(self,
                             prompt: str,
                             **kwargs
                             ):
        """
        Get logits over the entire vocabulary for the first token generated

        :param prompt: formatted prompt string
        :param kwargs: model.generation() parameters
        :return: logit tensor for the next token
        """

        output = self.get_completion(prompt, **kwargs)
        next_token_logits = output.scores[0]

        return next_token_logits

    # TODO: metaprompting
    def run_metaprompt(self):
        return
