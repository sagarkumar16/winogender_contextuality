from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import requests
import time
import yaml
from munch import munchify
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
import numpy as np
import utils as ut
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
    llm_params = {#"do_sample": False,
            "max_tokens": 8,
            'logprobs': True,
            'top_logprobs': 4
            }
else:
    llm_params = {#"do_sample": True,
            "temperature": temperature,
            #"top_k": 10,
            "max_tokens": 8,
            'logprobs': True,
            'top_logprobs': 4
            }

API_TOKEN = params.model.API_TOKEN
headers = {"Authorization": f"Bearer {API_TOKEN}", "x-use-cache": 'false'}
API_URL = "https://api-inference.huggingface.co/models/"+params.model.model_name
tokenizer = AutoTokenizer.from_pretrained(params.model.model_name, token=API_TOKEN)
client = InferenceClient(api_key=API_TOKEN, headers = headers)

def query(payload):
    try:
        response = client.chat.completions.create(
                    model=params.model.model_name,
                    messages=payload,
                    #response_format={'type': 'json'},
                    **llm_params
                ).choices[0]#.message.content
    except:
        logger.warning(f"Failed to query {payload}")
        return None
    return response


def api_hit(chat, options, first_target_phrase):
    """Generate a response from the model."""

    overloaded = 1
    while overloaded == 1:
        response = query(chat)  # query({"inputs": chat, "parameters": llm_params, "options": {"use_cache": False}})

        if response is None:
            logger.debug(f"Response JSON Error")
            print('CAUGHT JSON ERROR')
            continue


        if type(response) == dict:
            logger.warning(f"EXCEPTION: {response}")
            print("AN EXCEPTION: ", response)
            time.sleep(2.5)
            if "Inference Endpoints" in response['error']:
                logger.warning(f"Rate Limit Reached for {response['error']['request_id']}")
                print("HOURLY RATE LIMIT REACHED")
                time.sleep(450)
            continue

        try:
            outputs = [response.logprobs.content[i].top_logprobs for i in range(len(response.logprobs.content))]
            token_outputs = [[o.token for o in output] for output in outputs]
        except:
            logger.warning(f"No logprobs found for {response}.")
            continue

        if len([i for i in range(len(token_outputs)) if
                any(phrase == token_outputs[i][0] for phrase in first_target_phrase)]) != 0:
            overloaded = 0
        else:
            logger.warning("FIRST PHRASE NOT FOUND IN index 0 POSITION IN ANY TOKEN POSITION")
            logger.debug(f"Response: {response.message.content}")
            logger.debug("Output tokens:")
            for i, o in enumerate(token_outputs):
                token = o[0]
                match = any(phrase == token for phrase in first_target_phrase)
                logger.debug(f"[{i}] Token: '{token}' | Match: {match} | Full: {o}")

        # if any(option in response.message.content for option in options):
        #     overloaded=0

    return [response, outputs, token_outputs]


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


def get_meta_response(chat):
    """Generate a response from the Llama model."""

    overloaded = 1
    while overloaded == 1:
        response = query(chat)  # query({"inputs": chat, "parameters": llm_params, "options": {"use_cache": False}})
        # print(response)
        if response == None:
            print('CAUGHT JSON ERROR')
            continue

        if type(response) == dict:
            print("AN EXCEPTION")
            time.sleep(2.5)
            if "Inference Endpoints" in response['error']:
                print("HOURLY RATE LIMIT REACHED")
                time.sleep(900)


        elif len(response.message.content.split(";")) < 2:
            print(f"RESPONSE SPLIT: {response.message.content.split(';')}")
            overloaded = 1
        # if 'value' in response['generated_text']:
        #     overloaded=0

        #     response_split = response['generated_text'].split(";")
        #     response_split = response_split[0].split(": ")
        #     if len(response_split)<2:
        #         overloaded = 1
    response_split = response.message.content.split(";")
    print(response_split[0])
    # time.sleep(5)
    return response_split[0]


def encode_decode_options(options):
    target_phrase = [[tokenizer.decode(target_token_id, skip_special_tokens=True) for target_token_id in
                      tokenizer.encode(option, add_special_tokens=False)] for option in options]
    # target_phrase = [tokenizer.decode(target_id, skip_special_tokens=True) for target_id in target_ids]
    first_target_phrase = [target[0] for target in target_phrase]
    print(f"Options tokens: {target_phrase}")
    print(f"first options tokens: {first_target_phrase}")
    first_target_id_dict = {option: first_target_phrase[i] for i, option in enumerate(options)}
    return first_target_id_dict


def get_probability_dict(options, prompt, first_target_id_dict, temperature=params.params.temperature,
                         epsilon=np.finfo(float).eps):
    first_target_phrase = [first_target_id_dict[option] for option in options]
    response, outputs, token_outputs = api_hit(chat=prompt, options=options, first_target_phrase=first_target_phrase)
    probability_dict = {opt: -np.inf for opt in options}

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
        print(normed_log_probs)
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
