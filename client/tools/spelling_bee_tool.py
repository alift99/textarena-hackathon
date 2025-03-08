from itertools import product
import nltk
nltk.download('words')
from nltk.corpus import words
from langchain_core.tools import tool


valid_words = set(words.words())  # Load dictionary of words

@tool
def find_valid_word(char_list: list, n: int):
    """Find the first valid word using exactly n characters from the given list."""

    print(f'SpellingBee tool called: ```char_list: {char_list}, n: {n}```')

    output = []
    n_options = max(1, 10 - n)
    for perm in product(char_list, repeat=n):  # Generate permutations of length n
        word = ''.join(perm)
        if word in valid_words:
            output.append(word)
            if len(output) >= n_options:
                print(f'Tool output: {output}')
                return output
    print(f'Tool output: {output}')
    return output  # No valid word found
