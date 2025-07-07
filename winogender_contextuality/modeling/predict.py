from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import requests
import time
import yaml
from munch import munchify
from huggingface_hub import InferenceClient, ChatCompletionOutputLogprob
from transformers import AutoTokenizer
import numpy as np
import utils as ut
import gc
from winogender_contextuality.config import MODELS_DIR, PROCESSED_DATA_DIR
from winogender_contextuality.modeling.prompting import *
from winogender_contextuality.modeling.meta_prompting import *

"""
(Working description)
Returns model output probabilities and stores them in the data files. 

Adapted from: AI-names by Ariel Flint Ashery

Sagar Kumar, 2025
"""

# TODO: Fill in type annotations
# TODO: Finish replacing print with logger

app = typer.Typer()

with open("params.yaml", "r") as f:
    doc = yaml.safe_load(f)
params = munchify(doc)
# set temperature to 0 for deterministic outcomes
temperature = params.params.temperature
if temperature == 0:
    llm_params = {
            "max_tokens": 8,
            'logprobs': True,
            'top_logprobs': 4
            }
else:
    llm_params = {
            "temperature": temperature,
            "max_tokens": 8,
            'logprobs': True,
            'top_logprobs': 4
            }

API_TOKEN = params.model.API_TOKEN
headers = {"Authorization": f"Bearer {API_TOKEN}", "x-use-cache": 'false'}
API_URL = "https://api-inference.huggingface.co/models/"+params.model.model_name
tokenizer = AutoTokenizer.from_pretrained(params.model.model_name, token=API_TOKEN)
client = InferenceClient(api_key=API_TOKEN, headers=headers)

def query(payload: str) -> ChatCompletionOutputLogprob | None:
    """
    Uses InferenceClient to query the model given a properly formatted payload.

    :param payload: Prompt string properly formatted for the model
    :return: string in JSON format
    """

    try:

        response = client.chat.completions.create(
                    model=params.model.model_name,
                    messages=payload,
                    **llm_params
                ).choices[0]

    except: # Should we get rid of this generic exception?
        logger.error(f"Failed to query {payload}")
        return None

    return response

# TODO: chat for this and meta-prompting should be imported from files
def api_hit(chat,
            options,
            first_target_phrase):
    # first_target_phase can be dropped by just tokenizing options

    """Generate a response from the model."""

    overloaded = 1
    while overloaded == 1:
        response = query(chat)

        if response is None:
            logger.debug(f"Response JSON Error")
            print('CAUGHT JSON ERROR')
            continue

        if type(response) == dict:
            logger.warning(f"EXCEPTION: {response}")
            time.sleep(2.5)
            if "Inference Endpoints" in response['error']:
                logger.warning(f"Rate Limit Reached for {response['error']['request_id']}")
                print("HOURLY RATE LIMIT REACHED")
                time.sleep(450)
            continue

        try:
            lp = response.logprobs.content
            outputs = [lp[i].top_logprobs for i in range(len(lp))]
            token_outputs = [[o.token for o in output] for output in outputs]
        except:
            logger.warning(f"No logprobs found for {response}.")
            continue

        # make sure at least one option is present in 0 position
        # (prompt has been engineered to avoid "output probability leakage" to other positions)

        # TODO: should probabilities be normalized to ensure binary? Or is that cheating?
        if len([i for i in range(len(token_outputs)) if
                any(phrase == token_outputs[i][0] for phrase in first_target_phrase)]) != 0:
            overloaded = 0
        else:
            # first_target_phrase is probably the check to make sure the model answered correctly
            logger.warning("FIRST PHRASE NOT FOUND IN index 0 POSITION IN ANY TOKEN POSITION")
            logger.debug(f"Response: {response.message.content}")
            logger.debug("Output tokens:")
            for i, o in enumerate(token_outputs):
                token = o[0]
                match = any(phrase == token for phrase in first_target_phrase)
                logger.debug(f"[{i}] Token: '{token}' | Match: {match} | Full: {o}")

    return [response, outputs, token_outputs]

# TODO: write these
def run_local():
    import sys
    import torch
    from torch import cuda, bfloat16
    from transformers import AutoModelForCausalLM
    from transformers import BitsAndBytesConfig
    import transformers
    import bitsandbytes
    import accelerate
    import huggingface_hub

    # Check CUDA
    logger.info(f'torch available: {torch.cuda.is_available()}', flush=True)
    logger.info(torch.version.cuda)
    for i in range(torch.cuda.device_count()):
        logger.info(torch.cuda.get_device_properties(i))

    # Connect with Huggingface
    huggingface_hub.login(params.model.API_TOKEN)
    logger.info('Start', flush=True)

    return

# TODO: get rid of this
# Ok what the fresh hell is this
def encode_decode_options(options):
    # TODO: certainly there must an easier way than this
    target_phrase = [[tokenizer.decode(target_token_id, skip_special_tokens=True) for target_token_id in
                      tokenizer.encode(option, add_special_tokens=False)] for option in options]
    first_target_phrase = [target[0] for target in target_phrase]
    logger.info(f"Options tokens: {target_phrase}")
    logger.info(f"first options tokens: {first_target_phrase}")
    first_target_id_dict = {option: first_target_phrase[i] for i, option in enumerate(options)}
    return first_target_id_dict

# TODO: update to take in both API or local
@app.command()
def get_response(chat, options, first_target_phrase):
    response = api_hit(chat, options, first_target_phrase)[0]
    response_split = response.message.content.split("'")
    for opt in options:
        try:
            index = response_split.index(opt)
        except:
            continue
    # print(response_split[index])
    return response_split[index]

@app.command()
def get_meta_response(chat):
    """Generate a response from the Llama model."""

    overloaded = 1
    while overloaded == 1:
        response = query(chat)  # query({"inputs": chat, "parameters": llm_params, "options": {"use_cache": False}})
        # print(response)
        if response == None:
            logger.debug('CAUGHT JSON ERROR')
            continue

        if type(response) == dict:
            logger.debug("AN EXCEPTION")
            time.sleep(2.5)
            if "Inference Endpoints" in response['error']:
                logger.warning("HOURLY RATE LIMIT REACHED")
                time.sleep(900)


        elif len(response.message.content.split(";")) < 2:
            logger.info(f"RESPONSE SPLIT: {response.message.content.split(';')}")
            overloaded = 1
        # if 'value' in response['generated_text']:
        #     overloaded=0

        #     response_split = response['generated_text'].split(";")
        #     response_split = response_split[0].split(": ")
        #     if len(response_split)<2:
        #         overloaded = 1
    response_split = response.message.content.split(";")
    logger.info(response_split[0])
    return response_split[0]

# TODO: Store this in an external doc
# TODO: understand what Ariel is doing here with the management of where these answers exist

@app.command()
def get_probability_dict(options,
                         prompt,
                         first_target_id_dict,
                         epsilon=np.finfo(float).eps):
    # What is this dictionary? Is he hoping to see the sentence or the kw in the first output?
    # He's just looking for the word, but word-as-tokenized token
    # Can be simplified by just getting the token for output desired
    first_target_phrase = [tokenizer.convert_ids_to_tokens(tokenizer.encode(option, add_special_tokens=False))
                           for option in options]

    response, outputs, token_outputs = api_hit(chat=prompt, options=options, first_target_phrase=first_target_phrase)
    probability_dict = {opt: -np.inf for opt in options}

    # TODO: Determine if this is still necessary given the current prompts
    # find the token location where both options exist.
    ## find indices of these locations in each successive generation
    outputs = [response.logprobs.content[i].top_logprobs for i in range(len(response.logprobs.content))]
    token_outputs = [[o.token for o in output] for output in outputs]
    index_list = [i for i in range(len(token_outputs)) if
                  all(phrase in token_outputs[i] for phrase in first_target_phrase)]

    # print(f"Response: {response}")
    # print("Token outputs:")
    # for o in token_outputs:
    #     print(o)
    # now, we have the position of the token - let us take the probability!

    if len(index_list) == 0:
        index = \
        [i for i in range(len(token_outputs)) if any(phrase == token_outputs[i][0] for phrase in first_target_phrase)][
            0]
        # find the winning word.
        selected_option = options[
            [idx for idx in range(len(first_target_phrase)) if token_outputs[index][0] in first_target_phrase[idx]][0]]
        winning_prob = 0.0  # response.logprobs.content[index].top_logprobs[0].logprob
        probability_dict[selected_option] = winning_prob
        # return probability_dict

    else:
        for i, phrase in enumerate(first_target_phrase):
            # find index of option in vector
            try:
                index = token_outputs[index_list[0]].index(phrase)
            except:
                continue

            # find logprob
            selected_option = options[i]
            winning_prob = response.logprobs.content[index_list[0]].top_logprobs[index].logprob
            # print(response.logprobs.content[index_list[0]].top_logprobs[index].token,  np.exp(winning_prob))

            # if np.exp(winning_prob) < epsilon:
            #     winning_prob = -np.inf #np.log(epsilon)
            probability_dict[selected_option] = max(winning_prob, np.log(epsilon))

    # renormalize probability:
    options_log_probs = list(probability_dict.values())
    # print(index_list)

    if -np.inf in options_log_probs:
        # print("Token outputs:")
        # for o in token_outputs:
        #     print(o)
        # print(index_list)
        # print(options_log_probs, np.exp(options_log_probs))
        normed_probs = ut.normalize_probs(np.exp(options_log_probs))
        normed_log_probs = np.log(normed_probs)
        logger.info(normed_log_probs)
        # time.sleep(5)
    else:
        normed_log_probs = ut.normalize_logprobs(options_log_probs)

    for option, log_prob in zip(probability_dict.keys(), normed_log_probs):
        probability_dict[option] = log_prob
    # print(probability_dict)
    return probability_dict

# TODO: reformat to fit main() with typing CLI input

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
