import gc
import torch
import typer

app = typer.Typer()

@app.command()
def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()