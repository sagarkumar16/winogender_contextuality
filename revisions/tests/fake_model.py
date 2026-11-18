"""
A stand-in for ModelProbs that needs no GPU or weights, for running the real collection loops in
winogender_contextuality.modeling.collect_sequential.

- Token "ids" are strings, and decoding concatenates them, so the collectors' decoding logic runs
  unchanged.
- get_completion() reads the prompt back (passage and presented options) and answers through a
  `policy(passage, options, rng) -> list of chosen options`, one per blank.
- get_raw_logits() returns logit[t] = t, so stored logits reveal which token ids were looked up.
"""

import ast
import random
import re
from types import SimpleNamespace

import torch

from winogender_contextuality.modeling.ModelProbs import ModelProbs

PRONOUN_IDS = {"he": 11, "she": 12, "him": 13, "her": 14, "his": 15}
FEMALE = {"she", "her"}


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 0

    def __init__(self):
        self._other_ids = {}

    def encode(self, text, add_special_tokens=False):
        if text in PRONOUN_IDS:
            return [PRONOUN_IDS[text]]
        return [self._other_ids.setdefault(text, 100 + len(self._other_ids))]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(tokens)


class _Inputs:
    def __init__(self, n_tokens):
        self.shape = (1, n_tokens)


def parse_prompt(prompt: list[dict]) -> tuple[str, list[list[str]]]:
    """The passage and the presented option list(s) of a single- or two-blank prompt."""
    user = next(m["content"] for m in prompt if m["role"] == "user")
    passage = user.split("*")[1]
    options = [ast.literal_eval(o) for o in re.findall(r"one of the options: (\[[^\]]*\])", user)]
    return passage, options


def first_option_policy(passage, options, rng):
    """Always picks the option presented first."""
    return [o[0] for o in options]


class FakeModelProbs(ModelProbs):
    policy = staticmethod(first_option_policy)
    prompts = []
    instances = []

    def __init__(self, mode, model_name, key=None, model_path=None, quantized=True):
        self.mode = mode
        self.model_name = model_name
        self.key = key
        self.model_path = model_path
        self.quantized = quantized
        self.model = None
        self.tokenizer = FakeTokenizer()
        self.rng = random.Random(len(type(self).instances))
        type(self).instances.append(self)

    def load_model(self):
        pass

    def get_raw_logits(self, prompt):
        return torch.arange(1000, dtype=torch.float32)

    def get_completion(self, prompt, max_new_tokens=6, continue_final_message=True, **kwargs):
        type(self).prompts.append(prompt)
        passage, options = parse_prompt(prompt)
        answers = type(self).policy(passage, options, self.rng)

        if len(answers) == 1:
            opening, body = "{'BLANK':'", f"{answers[0]}'}}"
        else:
            opening, body = "{'BLANK1':'", f"{answers[0]}', 'BLANK2':'{answers[1]}'}}"

        has_prefill = continue_final_message and prompt[-1]["role"] == "assistant"
        prefill = prompt[-1]["content"] if has_prefill else ""
        # Without a prefill the model opens its own turn, as gemma's chat template does
        text = body if has_prefill else "model\n" + opening + body

        prompt_tokens = ["<system>", "<user>", "", "", "", "", prefill]
        return _Inputs(len(prompt_tokens)), SimpleNamespace(sequences=[prompt_tokens + list(text)])


def fake_model_probs(policy=first_option_policy):
    """A FakeModelProbs subclass with its own policy and its own prompt / instance logs."""
    return type("FakeModelProbsWithPolicy", (FakeModelProbs,),
                {"policy": staticmethod(policy), "prompts": [], "instances": []})
