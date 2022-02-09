import numpy as np
import csv
import emoji

def read_csv(filename):
    phrase = []
    emoji = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y

def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)

emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}


def read_glove_vecs(glove_file):
    with open(glove_file,'r') as f:
        words = set()
        word_to_vec_map = dict()
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:],dtype=np.float64)
        
        i = 1
        words_to_index = dict()
        index_to_words = dict()
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def convert_to_one_hot(Y,C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y