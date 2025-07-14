import gc
import torch
import typer
import torch.nn.functional as F

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