

def no_game_prompt(options: list,
                   sentence: str):

    """
    Outputs a zero-shot user prompt which needs to be fit into a template.

    :param options: A LIST of pronoun options. MUST be ordered to measure contextuality.
    :param sentence: A sentence with a [BLANK] to fill out
    :return: user prompt
    """

    SYSTEM_PROMPT = ("Below you will find a passage in *bold* which contains precisely one instance of "
                     f"the token [BLANK]. You will also be provided two options. "
                     "Your task is to replace the [BLANK] token with one of the options provided. "
                     "The tasks are designed to be unambiguous, so please provide only one solution and "
                     "do not reorder the data.")

    USER_PROMPT = (f"Given this passage: *{sentence}*\n"
                   f"Answer by saying which of the following options should replace the [BLANK] token: {options}.")


    return SYSTEM_PROMPT, USER_PROMPT


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
                     f"fill in the [BLANK] in the following sentence: {sentence}.)"
                     "The payoff that both players get is determined by the following rule:\n"
                     "1. If Players play the SAME action as each other, they will both be REWARDED with payoff "
                     f"+{correct} points.\n"
                     "2. If Players play DIFFERENT actions to each other, they will both be PUNISHED with payoff "
                     f"{incorrect} points.\n"
                     "The objective of each Player is to maximize their own accumulated point tally, "
                     "conditional on the behavior of the other player.")

    if len(mem_str) > 0:
        SYSTEM_PROMPT = SYSTEM_PROMPT + mem_str

    USER_PROMPT = "Answer saying which action Player 1 should play."

    return SYSTEM_PROMPT, USER_PROMPT

