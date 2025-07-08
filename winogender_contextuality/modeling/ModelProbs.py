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

        self.mode = mode
        self.model_name = model_name
        self.key = key


        # Local Run Attributes
        self.quantized = quantized
        self.model_path = model_path
        self.model = None


    # TODO: if API, use pipeline
    def load_model(self):

        if self.mode == 'api':
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
    def get_raw_logits(self):
        return

    # TODO: prompting generation method
    def get_completion(self):

        return

    # TODO: metaprompting
    def run_metaprompt(self):
        return
