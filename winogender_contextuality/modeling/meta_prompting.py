# %%
import random
import winogender_contextuality.modeling.prompting as pr
from winogender_contextuality.utils import *
from winogender_contextuality.config import *

# %%

# TODO: Adjust questions as necessary
# TODO: this should take in the model from a ModelProbs class

def get_question(q:str,
                 blank_role: str,
                 other_role: str):

    """

    :param q:
    :param blank_role:
    :param other_role:
    :return:
    """

    if q == 'anaphora':

        question = ("Answer saying who the pronoun replaced by BLANK is referring to. "
                    "Select from one of the following options: "
                    f"[{blank_role}, {other_role}].")

    if q == "pos":
        question = ("Answer saying what part of speech the BLANK should be. "
                    "Select from one of the following options: "
                    "[noun, verb, pronoun, adjective, adverb, preposition, article].")

    if q == "other_gender":
        question = (f"Answer saying the gender of the {other_role}. "
                    "Select from one of the following options: "
                    "[male, female]")
    
    return question

# TODO: update this
def get_meta_prompt_list(some_player, rules, options):
    q_list = ['min', 'max', 'actions', 'payoff', 'round', 'action_i', 'points_i', 'no_actions', 'no_points']
    if len(some_player['my_history']) == 0:
        q_list = ['min', 'max', 'actions', 'payoff', 'round']
        i = 1

    else:
        i = random.choice(range(len(some_player['my_history']))) + 1
    prompts = []
    questions = []
    for q in q_list:
        question = get_question(q, i, options)
        questions.append(question)
        prompts.append(pr.get_prompt(player=some_player, memory_size=len(some_player['my_history']), rules=rules,
                                     question=question))

    return i, questions, q_list, prompts


def get_answers(q,
                idx,
                role_dict):

    ## need genders for all of them
    ## need to both roles
    ## can only run other_gender where there is a priming sentence

    if q == 'anaphora':
        role_dict
        return

    if q == 'pos':
        return 'pronoun'

    if q == 'other_gender':
        return



def run_metaprompting():

    mp = ModelProbs(
        mode=mode,
        model_name=model_name,
        key=HF_KEY,
        model_path=MODELS_DIR,
        quantized=quantized
    )
    mp.load_model()

    role_dict = get_role_dict()



    # wrap in this
    prompt = pr.role_content_base()


