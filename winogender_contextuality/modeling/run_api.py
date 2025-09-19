import anthropic
from pathlib import Path
from typing import Any, Optional
import json
import os
from winogender_contextuality.config import *

class ClaudeModel:
    """
    A wrapper class that mimics the interface of a HuggingFace model
    but uses Claude API for inference.
    """
    
    def __init__(self, client: anthropic.Anthropic, model_name: str = "claude-sonnet-4-20250514"):
        self.client = client
        self.model_name = model_name
        self.config = type('Config', (), {'use_cache': False})()
    
    def generate(self, 
                 input_text: str, 
                 max_tokens: int = 1000, 
                 temperature: float = 0.7,
                 **kwargs) -> str:
        """
        Generate text using Claude API
        
        :param input_text: The input prompt
        :param max_tokens: Maximum tokens to generate
        :param temperature: Sampling temperature
        :return: Generated text
        """
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": input_text}
                ]
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Error generating text with Claude: {e}")
            raise
    
    def __call__(self, *args, **kwargs):
        """Make the model callable like HuggingFace models"""
        if args and isinstance(args[0], str):
            return self.generate(args[0], **kwargs)
        else:
            # Handle other calling patterns if needed
            return self.generate(**kwargs)


def load_model(model_name: str,
               api_key: str,
               quantized: bool,
               model_path: str) -> ClaudeModel:
    """
    Creates a Claude API client and returns a model-like wrapper.
    Maintains the same signature as the original function for compatibility.
    
    :param model_name: Claude model name (e.g., "claude-sonnet-4-20250514")
    :param api_key: Anthropic API key
    :param quantized: Ignored for Claude API (kept for compatibility)
    :param model_path: Directory for caching (used for config storage)
    :return: ClaudeModel wrapper
    """
    
    # Create cache directory for consistency
    cache_dir = Path(model_path) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate Claude model name or use default
    valid_claude_models = [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250513",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ]
    
    if model_name not in valid_claude_models:
        logger.warning(f"Model name '{model_name}' not in known Claude models. Using default.")
        model_name = "claude-sonnet-4-20250514"
    
    logger.info(f"Initializing Claude API with model: {model_name}", flush=True)
    
    # Initialize Anthropic client
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # Test the connection with a simple request
        test_message = client.messages.create(
            model=model_name,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        logger.info("Successfully connected to Claude API", flush=True)
        
    except Exception as e:
        logger.error(f"Failed to connect to Claude API: {e}", flush=True)
        raise
    
    # Create and return the wrapper model
    claude_model = ClaudeModel(client, model_name)
    
    # Save configuration for future reference
    config_path = cache_dir / "claude_config.json"
    config_data = {
        "model_name": model_name,
        "quantized": quantized,  # Stored for compatibility but not used
        "cache_dir": str(cache_dir)
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    logger.info(f"Claude model configuration saved to {config_path}", flush=True)
    logger.info(f"Note: 'quantized' parameter ({quantized}) is ignored for Claude API", flush=True)
    
    return claude_model


# Additional utility functions for compatibility

def load_tokenizer(model_name: str, api_key: str, cache_dir: Optional[str] = None):
    """
    Placeholder tokenizer function for compatibility.
    Claude API handles tokenization internally.
    """
    logger.warning("Tokenizer not needed for Claude API - tokenization handled internally")
    return None


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        exit(1)
    
    model_path = "./claude_models"
    
    # Load Claude model
    claude = load_model(
        model_name="claude-sonnet-4-20250514",
        api_key=api_key,
        quantized=False,  # Ignored for Claude
        model_path=model_path
    )
    
    # Test generation
    prompt = "Explain the concept of machine learning in one sentence."
    response = claude.generate(prompt, max_tokens=100, temperature=0.3)
    print(f"Response: {response}")