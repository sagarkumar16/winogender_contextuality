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
            inputs = self.tokenizer.apply_chat_template(prompt, return_tensors="pt", continue_final_message=True).to(self.gpu)
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

        #assert all([s[0] == " " for s in options]), "Tokens must begin with a space. "
        # -- no longer true in JSON output format

        token_ids = [self.tokenizer.encode(opt, add_special_tokens=False) for opt in options]

        for n, id in enumerate(token_ids):
            if len(id) > 1:
                logger.info(f"{options[n]} decomposes into two tokens.")

        return token_ids

    # TODO: prompting generation method
    #  - this will have to do all the first_id nonsense to make sure we are looking at the correct probs
    def get_completion(self,
                       prompt: str,
                       continue_final_message: bool = True,
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
            'max_new_tokens': 6,
            'temperature': 0.5,
            'top_k': 40
        }

        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        default_args["pad_token_id"] = pad_id if pad_id is not None else eos_id
        default_args["eos_token_id"] = eos_id

        # If kwargs were provided, update the defaults -- this is where MODEL_PARAMS.params from config go
        generation_args = {**default_args, **kwargs}

        if self.mode == 'api':
            raise NotImplementedError

        else:
            inputs = (self.tokenizer.apply_chat_template(prompt, return_tensors="pt",
                                                         continue_final_message=continue_final_message)
                      .to(self.gpu))

            outputs = self.model.generate(inputs, **generation_args)

        return inputs, outputs

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

        _, output = self.get_completion(prompt, **kwargs)
        next_token_logits = output.scores[0]

        return next_token_logits

    def find_pronouns(self,
                      pronouns: list,
                      generated_sequence: torch.Tensor,
                      scores: tuple[torch.Tensor] | None = None,
                      logits: bool = True
                      ) -> torch.Tensor | tuple[torch.Tensor]:


        """
        Find positions (and logits) for pronouns in generated text.

        :param pronouns: list of pronouns searching for
        :param generated_sequence: only the tokens generated
        :param scores: output scores for generated tokens, only needed if logits = True
        :param logits: whether to return logits tensors
        """

        trial_strings = pronouns + [' ' + p for p in pronouns]
        token_tensor = self.tokenizer(trial_strings + pronouns)
        possible_tokens = torch.tensor([t[1] for t in token_tensor.input_ids]).to('cuda:0')
        mask = (generated_sequence[..., None] == possible_tokens).any(-1)
        indices = torch.nonzero(mask, as_tuple=False)

        if logits:
            if scores is None:
                raise TypeError("Output scores required to output logits.")
            else:
                output_logits = scores[indices.item()]
                return indices, output_logits

        else:
            return indices

    def pronoun_logits(self,
                       pronouns_list: list[list[str]],
                       generated_sequence: torch.Tensor,
                       scores: tuple[torch.Tensor] | None = None
                      ) -> torch.Tensor | tuple[torch.Tensor]:

        sep_token = self.tokenizer('BLANK2').input_ids[1]
        
        token_log_dict = {g.cpu().item(): arr for g, arr in zip(generated_sequence, scores)}
        idx = list(token_log_dict.keys()).index(sep_token)
    
        first_idx, first_logits = self.find_pronouns(pronouns_list[0], generated_sequence[:idx], scores[:idx], logits=True)
        second_idx, second_logits = self.find_pronouns(pronouns_list[1], generated_sequence[idx+1:], 
                                      scores[idx+1:], logits=True)
    
        first_token = self.tokenizer.convert_ids_to_tokens([generated_sequence[:idx][first_idx.item()]])
        second_token = self.tokenizer.convert_ids_to_tokens([generated_sequence[idx+1:][second_idx.item()]])
    
        return (first_token, first_logits, second_token, second_logits)

    # TODO: metaprompting
    def run_metaprompt(self):
        return
