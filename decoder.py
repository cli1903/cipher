import re
import math
import random
import time
from collections import Counter

random.seed(0)
ALPHABET = 'abcdefghijklmnopqrstuvwxyz '
MINING_FILE = 'les-miserables.txt'
ENCODED_FILE_1 = 'h_19.txt'
ENCODED_FILE_2 = 'j_19.txt'
ENCODED_FILE_3 = 'f_19.txt'


def clean_text(file_path):
    """Reads in and cleans text to be used for true language mining.

    Reads in text, sets all letters to lowercase and removes characters not in
    our alphabet.

    Args:
        file_path: The path to the text.

    Returns:
        The cleaned up data.
    """
    with open(f'./texts/{file_path}') as f:
        text = f.read().lower().replace('\n', ' ')
    text = re.sub(r'[^a-z\s]', '', text)
    return text


def compute_probs(text):
    """Computes the probabilities of each character in our alphabet.

    Mines the probabilities of each character in the alphabet based on text by
    calculating the empirical probabilities.

    Args:
        text: The cleaned up text, containing only lowercase chars in our alphabet.

    Returns:
        A dict mapping chars in our alphabet to its probability.
    """
    char_counts = Counter(text)
    total_chars = len(text)
    for k, v in char_counts.items():
        char_counts[k] = v / total_chars
    return char_counts


def compute_cond_probs(text):
    """Computes the conditional probabilities of each char pair in our alphabet.

    Mines the conditional probabilities of each pair of characters in the
    alphabet based on text by calculating the empirical probabilities.

    Args:
        text: The cleaned up text, containing only lowercase chars in our alphabet.

    Returns:
        A dict mapping chars i in our alphabet to a dict mapping chars j in our
        alphabet to the conditional probability of char j given char i.
        ex: {'a': {'b': 0.5, 'c': 0.5}, 'b': {'d': 1.0}}
    """
    pair_counts = Counter(zip(text[:-1], text[1:]))
    trailing_char_counts = Counter(text)
    # all but the last char has another char following it
    trailing_char_counts[text[-1]] -= 1
    cond_probs = {}
    for (i, j), v in pair_counts.items():
        if i in cond_probs:
            cond_probs[i][j] = v / trailing_char_counts[i]
        else:
            cond_probs[i] = {j: v / trailing_char_counts[i]}
    return cond_probs


def decode_string(text, decoder):
    return ''.join([decoder[c] for c in text])


def energy(text, decoder, probs, cond_probs):
    """Computes the 'energy' for a specific encoding.

    Computes the energy function, or the -log of the likelihood function, based
    on the decoder and mined probabilities.

    Args:
        text: The cleaned up text to apply the decoder to, containing only
          lowercase chars in our alphabet.
        decoder: A dict mapping char to char, how to decode the text
        probs: A dict mapping each char in our alphabet to its probability.
        cond_probs: A dict mapping each char i to a dict that maps char j to the
          conditional probability of char j given char i.
          ex: {'a': {'b': 0.5, 'c': 0.5}, 'b': {'d': 1.0}}

    Returns:
        The -log of the likelihood function of the specified decoder given the
        encoded text.
    """
    decoded_text = decode_string(text, decoder)
    energy = 0
    for i, c in enumerate(decoded_text):
        if i == len(decoded_text) - 1:
            continue
        if i == 0:
            energy -= math.log(probs[c])
        # to avoid log(0), return 0.000001 rather than 0 if pair does not exist
        energy -= math.log(cond_probs[c].get(decoded_text[i+1], 0.000001))
    return energy


def metropolis(encoded_text, probs, cond_probs, beta=1, n_iters=10**6, print_text=False):
    """Runs a Metropolis scheme to decode a text.

    Runs a Metropolis scheme for specified number of iterations to decode a text
    using the given probabilities.

    Args:
        text: The encoded cleaned up text, containing only lowercase chars in
          our alphabet.
        probs: A dict mapping each char in our alphabet to its probability.
        cond_probs: A dict mapping each char i to a dict that maps char j to the
          conditional probability of char j given char i.
          ex: {'a': {'b': 0.5, 'c': 0.5}, 'b': {'d': 1.0}}
        beta: parameter for determining when to transition to proposed decoder
        n_iters: the number iterations to run the algorithm for

    Returns:
        The decoder the text represented by a dict mapping char to char.
    """
    counter = 0
    previous_swap = 0
    x_decoder = dict(zip(ALPHABET, ALPHABET))
    while counter < n_iters:
        if print_text and counter % 100 == 0:
            print(counter, decode_string(encoded_text, x_decoder))
            to_continue = input('continue algorithm?')
            if to_continue == 'no':
                return x_decoder
        y_decoder = x_decoder.copy()
        swap1 = random.choice(ALPHABET)
        swap2 = random.choice(ALPHABET)
        while swap2 == swap1:  # makes sure that swap1 and swap2 are diff chars
            swap2 = random.choice(ALPHABET)
        y_decoder[swap1], y_decoder[swap2] = y_decoder[swap2], y_decoder[swap1]
        y_energy = energy(encoded_text, y_decoder, probs, cond_probs)
        x_energy = energy(encoded_text, x_decoder, probs, cond_probs)
        delta = y_energy - x_energy
        if delta < 0 or random.random() < math.exp(-beta * delta):
            x_decoder = y_decoder
            #print(counter - previous_swap)
            previous_swap = counter
        counter += 1
    return x_decoder

def metropolis_early_stop(encoded_text, probs, cond_probs, beta=0.5, stop_n=100):
    """Runs a Metropolis scheme to decode a text.

    Runs a Metropolis scheme for specified number of iterations to decode a text
    using the given probabilities.

    Args:
        text: The encoded cleaned up text, containing only lowercase chars in
          our alphabet.
        probs: A dict mapping each char in our alphabet to its probability.
        cond_probs: A dict mapping each char i to a dict that maps char j to the
          conditional probability of char j given char i.
          ex: {'a': {'b': 0.5, 'c': 0.5}, 'b': {'d': 1.0}}
        beta: parameter for determining when to transition to proposed decoder
        stop_n: how many iterations the decoder has to remain the same before
          the algorithm can terminate

    Returns:
        The decoder for the text represented by a dict mapping char to char.
    """
    counter = 0
    n_iters = 0
    x_decoder = dict(zip(ALPHABET, ALPHABET))
    while counter <= stop_n:
        counter += 1
        y_decoder = x_decoder.copy()
        swap1 = random.choice(ALPHABET)
        swap2 = random.choice(ALPHABET)
        while swap2 == swap1:  # makes sure that swap1 and swap2 are diff chars
            swap2 = random.choice(ALPHABET)
        y_decoder[swap1], y_decoder[swap2] = y_decoder[swap2], y_decoder[swap1]
        y_energy = energy(encoded_text, y_decoder, probs, cond_probs)
        x_energy = energy(encoded_text, x_decoder, probs, cond_probs)
        delta = y_energy - x_energy
        if delta < 0 or random.random() < math.exp(-beta * delta):
            x_decoder = y_decoder
            counter = 0
        n_iters += 1
    return x_decoder, n_iters


def main():
    cleaned_mining_text = clean_text(MINING_FILE)
    cleaned_encoded_1 = clean_text(ENCODED_FILE_1)
    cleaned_encoded_2 = clean_text(ENCODED_FILE_2)
    cleaned_encoded_3 = clean_text(ENCODED_FILE_3)

    probs = compute_probs(cleaned_mining_text)
    cond_probs = compute_cond_probs(cleaned_mining_text)

    start = time.time()
    #decoder1 = metropolis(cleaned_encoded_1, probs, cond_probs, n_iters=10**4)
    #decoder1 = metropolis(cleaned_encoded_1, probs, cond_probs)
    decoder1 = metropolis(cleaned_encoded_1, probs, cond_probs, beta=0.5, n_iters=10**4)
    #decoder1 = metropolis(cleaned_encoded_1, probs, cond_probs, beta=0.5, n_iters=10**4, print_text=True)
    #decoder1, n1 = metropolis_early_stop(cleaned_encoded_1, probs, cond_probs)
    #print(n1)
    with open(f'./decoded/decoded_{ENCODED_FILE_1}', 'w') as f:
        f.write(decode_string(cleaned_encoded_1, decoder1))

    #decoder2 = metropolis(cleaned_encoded_2, probs, cond_probs, n_iters=10**4)
    #decoder2 = metropolis(cleaned_encoded_2, probs, cond_probs)
    decoder2 = metropolis(cleaned_encoded_2, probs, cond_probs, beta=0.5, n_iters=10**4)
    #decoder2 = metropolis(cleaned_encoded_2, probs, cond_probs, beta=0.5, n_iters=10**4, print_text=True)
    #decoder2, n2 = metropolis_early_stop(cleaned_encoded_2, probs, cond_probs)
    #print(n2)
    with open(f'./decoded/decoded_{ENCODED_FILE_2}', 'w') as f:
        f.write(decode_string(cleaned_encoded_2, decoder2))

    #decoder3 = metropolis(cleaned_encoded_3, probs, cond_probs, n_iters=10**4)
    #decoder3 = metropolis(cleaned_encoded_3, probs, cond_probs)
    decoder3 = metropolis(cleaned_encoded_3, probs, cond_probs, beta=0.5, n_iters=10**4)
    #decoder3 = metropolis(cleaned_encoded_3, probs, cond_probs, beta=0.5, n_iters=10**4, print_text=True)
    #decoder3, n3 = metropolis_early_stop(cleaned_encoded_3, probs, cond_probs)
    #print(n3)
    with open(f'./decoded/decoded_{ENCODED_FILE_3}', 'w') as f:
        f.write(decode_string(cleaned_encoded_3, decoder3))
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()
