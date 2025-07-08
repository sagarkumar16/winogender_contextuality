from transformers import AutoTokenizer
import torch
from winogender_contextuality.modeling.run_local import load_model
from winogender_contextuality.utils import flush

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
                 quantized: bool = True,
                 ):

        self.mode = mode == 'api' # self.mode = True maps to API call instead of local
        self.model_name = model_name
        self.key = key
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=key)


        # Local Run Attributes
        self.quantized = quantized
        self.model_path = model_path
        self.model = None


    # TODO: if API, use pipeline
    def load_model(self) -> None:

        """
        Loads a huggingface pretrained model locally, sets that as self.model, and caches it in the model_path directory
        :return: None
        """

        if self.mode:
            raise NotImplementedError

        elif self.mode == 'gpu':
            if self.model_path is None:
                raise AttributeError("Must specify a model path for local runs")
            else:
                flush()
                self.model = load_model(self.model_name, self.key, self.quantized, self.model_path)

        else:
            raise AttributeError("Mode must be either 'api' or 'gpu'")

    # TODO: prompting raw method
    def get_raw_logits(self,
                       prompt: str):

        """
        Obtains raw logits for the entire vocabulary of the model
        :param prompt:
        :return:
        """

        inputs = (self.tokenizer.apply_chat_template(prompt, return_tensors="pt", continue_final_message=True)
                  .to("cuda:0"))

        if self.mode:
            raise NotImplementedError

        else:
            with torch.no_grad():
                outputs = self.model(inputs)

            logits = outputs.logits
            next_token_logits = logits[0, -1]

        return next_token_logits

    # TODO: prompting generation method
    #   this will have to do all the first_id nonsense to make sure we are looking at the correct probs
    def get_completion(self):
        return

    # TODO: masked softmax over selected tokens

    # TODO: softmax over all vocabulary (& plot?)

    # TODO: metaprompting
    def run_metaprompt(self):
        return
