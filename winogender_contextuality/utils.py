import gc
import torch
import typer
import torch.nn.functional as F
import ast

app = typer.Typer()

@app.command()
def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

def masked_softmax(token_ids: list[int],
                   logits: torch.Tensor) -> torch.Tensor:
    """
    Calculates probabilities based on a masked softmax which only considers the tokens in token_ids so that the sum
    of their probabilities is equal to 1.

    :param token_ids: list of selected token IDs
    :param logits: logits tensor from get_raw_logits() or get_completed_logits()
    :return: tensor containing normalized probabilities
    """

    z = logits[token_ids]
    probs = F.softmax(z, dim=0)

    return probs

def reverse_pronouns(options: str, 
                     measurement: str,
                     prime: str = "she_first") -> list[str]:

    """
    Takes in a pronoun ist string from the processed TSV file and returns the pronouns as a list in the correct order.
    
    :param options: list stored as a string (e.g. "["he", "she"]")
    :param measurement: measurement context (e.g. "he_first")
    :param prime: which measurement context needs to be reversed
    """

    pronouns = ast.literal_eval(options)
    if measurement==prime:
        return pronouns[::-1]
    else:
        return pronouns

    