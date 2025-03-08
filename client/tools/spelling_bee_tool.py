from itertools import product
import nltk
# nltk.download('words')
from nltk.corpus import words
from langchain_core.tools import tool


valid_words = set(words.words())  # Load dictionary of words

@tool
def find_valid_word(char_list: list, n: int):
    """Find the first valid word using exactly n characters from the given list."""

    print(f'SpellingBee tool called: ```char_list: {char_list}, n: {n}```')

    output = []
    n_options = 10
    # Iterate over valid_words and check if the word satisfies the conditions
    for word in valid_words:
        # Check if the word has exactly 'n' characters and only uses characters from 'char_list'
        if len(word) == n and all(char in char_list for char in word):
            output.append(word)
            if len(output) >= n_options:
                print(f'Tool output: {output}')
                return output
    print(f'Tool output: {output}')
    return output  # No valid word found
