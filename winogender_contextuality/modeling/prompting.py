
# NOTE THAT A SINGLE QUOTATION IS ADDED AT THE BLANKS

def no_game_prompt(options: list,
                   sentence: str,
                   system: bool = True,
                   user: bool = True,
                   assistant: bool = True):

    """
    Outputs a zero-shot user prompt which needs to be fit into a template.

    :param options: A LIST of pronoun options. MUST be ordered to measure contextuality.
    :param sentence: A sentence with a [BLANK] to fill out
    :param system: If true, a system prompt will be output
    :param user: If true, a user prompt will be output
    :param assistant: If true, a assistant prompt will be output
    :return: user prompt
    """

    SYSTEM_PROMPT = ("Below you will find a passage in *bold* which contains precisely one instance of "
                     f"the token BLANK. You will also be provided two options. "
                     "Your task is to replace the BLANK token with one of the options provided. "
                     "The tasks are designed to be unambiguous, so please provide only one solution and "
                     "do not reorder the data.")

    USER_PROMPT = (f"Given this passage: *{sentence}*\n"
                   f"Replace the BLANK with one of the following options: {options}.")

    ASSISTANT_PROMPT = sentence.split('BLANK')[0][:-1]


    return SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT

def no_game_seq_prompt(option_sets: list[list[str]],
                       sentences: list[str]):

    """
    Outputs a zero-shot user prompt which needs to be fit into a template. This prompt should be used with a generation head. 

    :param option_sets: A LIST of LIST of pronoun options. MUST be ordered to measure contextuality.
    :param sentences: Ordered list of two sentences, each with a BLANK
    :return: user prompt
    """

    sent1, sent2 = sentences
    sent1 = sent1.replace('BLANK', 'BLANK1')
    sent2 = sent2.replace('BLANK', 'BLANK2')

    SYSTEM_PROMPT = ("Below you will find a passage in *bold* which contains precisely one instance of "
                     "the term BLANK1, and one instance of the term BLANK2. "
                     "Your task is to replace BLANK1 and BLANK2 with one of the options provided for each. "
                     "The task is designed to be unambiguous, so please provide only one token for each blank and "
                     "do not reorder the data. Do not repeat the sentence.")

    USER_PROMPT = (f"Given this passage: *{sent1} {sent2}*\n" 
                   f"Replace BLANK1 with one of the options: {option_sets[0]}. " 
                   f"Replace BLANK2 with one of the options: {option_sets[1]}. "
                   "Respond only in the following format {'BLANK1': '<text>', 'BLANK2': '<text>'}"
                   )

    ASSISTANT_PROMPT = "{'BLANK1':'"

    return SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT

def no_game_seq_prompt(option_sets: list[list[str]],
                       sentences: list[str]):

    """
    Outputs a zero-shot user prompt which needs to be fit into a template. This prompt should be used with a generation head.

    :param option_sets: A LIST of LIST of pronoun options. MUST be ordered to measure contextuality.
    :param sentences: Ordered list of two sentences, each with a BLANK
    :return: user prompt
    """

    sent1, sent2 = sentences
    sent1 = sent1.replace('BLANK', 'BLANK1')
    sent2 = sent2.replace('BLANK', 'BLANK2')

    SYSTEM_PROMPT = ("Below you will find a passage in *bold* which contains precisely one instance of "
                     "the term BLANK1, and one instance of the term BLANK2. "
                     "Your task is to replace BLANK1 and BLANK2 with one of the options provided for each. "
                     "The task is designed to be unambiguous, so please provide only one token for each blank and "
                     "do not reorder the data. Do not repeat the sentence.")

    USER_PROMPT = (f"Given this passage: *{sent1} {sent2}*\n" 
                   f"Replace BLANK1 with one of the options: {option_sets[0]}. " 
                   f"Replace BLANK2 with one of the options: {option_sets[1]}. "
                   "Respond only in the following format {'BLANK1': '<text>', 'BLANK2': '<text>'}"
                   )

    ASSISTANT_PROMPT = "{'BLANK1':'"

    return SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT

#def no_game_seq_logit_prompt(option_set: list[str],
#                             free_sentence: str,
#                             fixed_sentence: None | str):
#
#    """
#    Outputs a zero-shot user prompt which needs to be fit into a template. This prompt should only be used to probe model state. 
#
#    :param option_set: A LIST of pronoun options. MUST be ordered to measure contextuality.
#    :param free_sentence: Sentence with a token BLANK in it. 
#    :param fixed_sentence: Additional sentence to test the effect of adding context.
#    :return: user prompt
#    """
#
#    SYSTEM_PROMPT = ("Below you will find a passage in *bold* which contains precisely one instance of "
#                     "the term BLANK. "
#                     "Your task is to replace BLANK with one of the options provided. "
#                     "The task is designed to be unambiguous, so please provide only one token for the blank and "
#                     "do not reorder the data.")
#    
#    if fixed_sentence:
#        sentence = fixed_sentence + " " + free_sentence
#    else:
#        sentence = free_sentence
#
#
#    USER_PROMPT = (f"Given this passage: *{sentence}*\n" 
#                       f"Replace BLANK with one of the options: {option_set}. "
#                       )
#
#    ASSISTANT_PROMPT = sentence.split('BLANK')[0][:-1]
#
#    return SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT

def no_game_seq_logit_prompt(option_set: list[str],
                             free_sentence: str,
                             fixed_sentence: None | str,
                             system: bool = True,
                             user: bool = True,
                             assistant: bool = True
                             ):

    """
    Outputs a zero-shot user prompt which needs to be fit into a template. This prompt should only be used to probe model state. 

    :param option_set: A LIST of pronoun options. MUST be ordered to measure contextuality.
    :param free_sentence: Sentence with a token BLANK in it. 
    :param fixed_sentence: Additional sentence to test the effect of adding context.
    :param system: If true, a system prompt will be output
    :param user: If true, a user prompt will be output
    :param assistant: If true, a assistant prompt will be output
    :return: user prompt
    """

    if system:
        SYSTEM_PROMPT = ("Below you will find a passage in *bold* which contains precisely one instance of "
                         "the term BLANK. "
                         "Your task is to replace BLANK with one of the options provided. "
                         "The task is designed to be unambiguous, so please provide only one token for the blank and "
                         "do not reorder the data. Do not repeat the sentence.")
    else:
        SYSTEM_PROMPT = None
    
    if fixed_sentence:
        sentence = fixed_sentence + " " + free_sentence
    else:
        sentence = free_sentence

    if user:
        USER_PROMPT = (f"Given this passage: *{sentence}*\n" 
                       f"Replace BLANK with one of the options: {option_set}. "
                       "Respond only in the following format {'BLANK': '<text>'}")
    else:
        USER_PROMPT = None

    if assistant:
        ASSISTANT_PROMPT = "{'BLANK':'"
    else:
        ASSISTANT_PROMPT = None

    return SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT


def no_game_seq_norder_prompt(option_set: list[str],
                             free_sentence: str,
                             fixed_sentence: None | str):
    """
    Outputs a zero-shot user prompt which needs to be fit into a template. This prompt should only be used to probe model state.

    :param option_set: A LIST of pronoun options. MUST be ordered to measure contextuality.
    :param free_sentence: Sentence with a token BLANK in it.
    :param fixed_sentence: Additional sentence to test the effect of adding context.
    :return: user prompt
    """

    SYSTEM_PROMPT = ("Below you will find a passage in *bold* which contains precisely one instance of "
                     "the term BLANK. "
                     "Your task is to replace BLANK with one of the options provided. "
                     "The task is designed to be unambiguous, so please provide only one token for the blank and "
                     "do not reorder the data. Do not repeat the sentence. Ignore the order of the options.")

    if fixed_sentence:
        sentence = fixed_sentence + " " + free_sentence
    else:
        sentence = free_sentence

    USER_PROMPT = (f"Given this passage: *{sentence}*\n"
                   f"Replace BLANK with one of the options: {option_set}. "
                   "Respond only in the following format {'BLANK': '<text>'}")

    ASSISTANT_PROMPT = "{'BLANK':'"

    return SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT

# TODO: Does this need an assistant prompt?
def game_prompt(options: list,
                sentence: str,
                rewards: list,
                mem_str: str = ""):

    """
    Outputs a role-play prompt which needs to be fit into a template.
    Memory string must be externally formatted before

    :param options: A LIST of pronoun options. MUST be ordered to measure contextuality.
    :param sentence: A sentence with a [BLANK] to fill out
    :param rewards: A LIST of rewards ordered as [incorrect, correct]
    :param mem_str: A string encoding the memory of past interactions
    :return: System prompt
    """

    incorrect, correct = rewards
    SYSTEM_PROMPT = ("Context: Player 1 is playing a multi-round partnership game with Player 2 for 100 rounds."
                     f"At each round, Player 1 and Player 2 simultaneously pick an action from the values {options} to "
                     f"fill in the BLANK in the following sentence: {sentence}.)"
                     "The payoff that both players get is determined by the following rule:\n"
                     "1. If Players play the SAME action as each other, they will both be REWARDED with payoff "
                     f"+{correct} points.\n"
                     "2. If Players play DIFFERENT actions to each other, they will both be PUNISHED with payoff "
                     f"{incorrect} points.\n"
                     "The objective of each Player is to maximize their own accumulated point tally, "
                     "conditional on the behavior of the other player.")

    if len(mem_str) > 0:
        SYSTEM_PROMPT = SYSTEM_PROMPT + " " + mem_str

    USER_PROMPT = "Answer saying which action Player 1 should play."

    ASSISTANT_PROMPT = "Player 1 should play {BLANK:'"

    return SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT

def role_content_base(system: str | None,
                      user: str | None,
                      assistant:str | None):

    if system is not None:
        system_dict = {
                "role": "system",
                "content": system
            }
    else:
        system_dict = None

    if user is not None:
        user_dict = {
                "role": "user",
                "content": user
            }
    else:
        user_dict = None

    if assistant is not None:
        assistant_dict = {
                "role": "assistant",
                "content": assistant
            }
    else:
        assistant_dict = None

    message = [
        system_dict,
        user_dict,
        assistant_dict
    ]

    return message

def get_role_content_prompt(game: bool,
                            options: list,
                            sentence: str,
                            rewards: list = [],
                            mem_str: str = ""):

    if game:
        assert rewards, "Game prompt requires rewards"
        SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT = game_prompt(options=options, sentence=sentence,
                                                                   rewards=rewards, mem_str=mem_str)
    else:
        SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT = no_game_prompt(options=options, sentence=sentence)

    message = role_content_base(SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT)

    return message
