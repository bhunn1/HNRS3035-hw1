import argparse
import pickle
import os
import re
import random

class NGram:
    """
    Class NGram provides the functionality to train a unigram or bigram model on a string of data and to make a
    prediction of words based on that trained model.

    :attribute n: 1 or 2, determines which model to use

    """

    def __init__(self, n: int):
        self.n = n
        self.dict_of_probs = {}

    def train(self, data: str):
        """
        For a given n (1 or 2), trains a unigram or bigram model on a string of data by:
        1. Forming the string into individual tokens
        2. Calculating the probability distribution (stored in self.dict_of_probs)

        :param data: the string to train

        """

        corpus = re.findall(r'\w+|[.,;!?]', data)

        # Unigram model
        if self.n == 1:
            tokens = list(dict.fromkeys(corpus))    # Creates a list of the unique words from the corpus

            # Store the occurences of every word and store the count that a word follows another
            occurences = {}
            count_follow = {}
            for i in range(len(corpus) - 1):
                occurences[corpus[i]] = occurences.get(corpus[i], 0) + 1

                t_i, t_j = corpus[i+1], corpus[i]
                if (t_i, t_j) not in count_follow:
                    count_follow[(t_i, t_j)] = 0
                count_follow[(t_i, t_j)] += 1

            # Loop through the tokens and store into the class attr the probability distribution
            for i in tokens:
                for j in tokens:
                    if (j, i) in count_follow.keys():
                        self.dict_of_probs[(j, i)] = count_follow[(j, i)] / occurences[j]

        # Bigram model
        elif self.n == 2:
            tokens = list(dict.fromkeys(corpus))

            # Same as unigram except occurences and count_follow store a sequence of two tokens
            occurences = {}
            count_follow = {}
            for i in range(len(corpus) - 2):
                occurences[(corpus[i], corpus[i+1])] = occurences.get((corpus[i], corpus[i+1]), 0) + 1

                t_i, s_j = corpus[i+2], (corpus[i], corpus[i+1])
                if (t_i, s_j) not in count_follow:
                    count_follow[(t_i, s_j)] = 0
                count_follow[(t_i, s_j)] += 1

            # Loops through the tokens and stores the probability distribution of a word following two words
            for (i, j), count in occurences.items():
                for next_word in tokens:
                    if (next_word, (i, j)) in count_follow:
                        self.dict_of_probs[(next_word, (i, j))] = count_follow[(next_word, (i, j))] / count

            print(self.dict_of_probs)

    def predict_next_word(self, input: tuple, nwords: int, deterministic: bool=False):
        """
        For a given n (1 or 2), predicts the next word(s) based on the provided trained model.

        :param input: tuple containing one or two words
        :param nwords: the number of words to predict
        :param deterministic: boolean flag that defaults to False. If true, method samples the highest probability next
        word. If false, the method samples a random word from the probability distribution.

        :return: a list of the words predicted

        """

        prediction = []

        # For prediction on unigram
        if self.n == 1:
            next_word_to_predict = input[0]

            for i in range(nwords):
                next_word_probs = {}

                # Stores the probability values of each word that follows the next word to predict
                for k, v in self.dict_of_probs.items():
                    if k[1] == next_word_to_predict:
                        next_word_probs[k[0]] = v

                # If the input/next_word is not in the prob distribution the loop will end
                if not next_word_probs:
                    print(f"{next_word_to_predict} not in vocabulary. Cannot find a valid next word. Stopping early.")
                    break

                if deterministic:
                    next_word = max(next_word_probs, key=next_word_probs.get)
                else:
                    next_word = random.choices(list(next_word_probs.keys()), weights=next_word_probs.values(), k=1)[0]

                prediction.append(next_word)
                next_word_to_predict = next_word    # Changes the next word to predict to the word obtained from calculation

        # For prediction on bigram, basically the same as unigram except next_word_to_predict is always a tuple of len 2
        elif self.n == 2:
            next_word_to_predict = (input[0], input[1])

            for i in range(nwords):
                next_word_probs = {}

                for k, v in self.dict_of_probs.items():
                    if k[1] == next_word_to_predict:
                        next_word_probs[k[0]] = v

                if not next_word_probs:
                    print("Cannot find a valid next word. Stopped early.")
                    break

                if deterministic:
                    next_word = max(next_word_probs, key=next_word_probs.get)
                else:
                    next_word = random.choices(list(next_word_probs.keys()), weights=next_word_probs.values(), k=1)[0]

                prediction.append(next_word)
                next_word_to_predict = (next_word_to_predict[1], next_word)
        print(prediction)
        return prediction

class BPEAlg:
    """
    Class BPEAlg implements a byte-pair encoding algorithm to train a model and provides functionality to predict words
    based on the trained model.

    :attribute vocabulary: stores the probability distribution as a list

    """

    def __init__(self):
        self.vocabulary = []

    def train(self, data: str, k: int=3000):
        """
        Trains a model on data to learn a vocabulary of language elements by merging frequent pairs of tokens and stores
        the vocabulary as a class attribute as a list

        :param data: the string to train
        :param k: the number of iterations for the BPE loop to run

        """

        tokens = re.findall(r'.', data)
        vocab = list(dict.fromkeys(tokens))     # Store the unique values from the list of tokens

        for i in range(k):
            print(f"Loading {((i+1)/k)*100}%...")
            pairs = []
            pair_freq = {}

            # Stores all pairs of characters from tokens
            for i in range(len(tokens) - 1):
                if tokens[i] != ' ':
                    pairs.append(tokens[i] + tokens[i+1])

            # Stores the frequency of pairs as a dict
            for i in range(len(pairs)):
                pair_freq[pairs[i]] = pair_freq.get(pairs[i], 0) + 1

            highest_freq_pair = max(pair_freq, key=pair_freq.get)

            replaced_tokens = []
            i = 0
            # Loop that creates a new list of tokens that combines the highest frequency pairs
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] + tokens[i+1] == highest_freq_pair:
                    replaced_tokens.append(highest_freq_pair)
                    i+=2
                else:
                    replaced_tokens.append(tokens[i])
                    i+=1

            tokens = replaced_tokens
            vocab.append(highest_freq_pair)

        self.vocabulary = vocab
        print(self.vocabulary)

    def tokenize(self, text: str) -> tuple:
        """
        Tokenizes a string based on the trained data provided by the BPE model

        :param text: the string to tokenize

        :return: a tuple of tokens and token IDs

        """

        split_text = re.findall(r'.', text)
        v = sorted(self.vocabulary, key=len, reverse=True)  # Sorts the vocabulary by length in descending order
                                                            # This helps to avoid incorrectly tokenizing smaller values
                                                            # when a bigger one exists
        for t in v:

            i = 0
            # Joins together the split text if any portion of it matches a token
            while i < len(split_text):
                if ''.join(split_text[i:i+len(t)]) == t:
                    split_text[i:i+len(t)] = [t]
                i += 1

        t_IDs = []

        # Stores the index of a token from the class vocabulary
        for t in split_text:
            if t in self.vocabulary:
                t_IDs.append(self.vocabulary.index(t))

        print(split_text, t_IDs)
        return split_text, t_IDs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="N-Gram Training",
                                     description="Train an NGram model and BPE Algorithm")
    parser.add_argument('s', choices=["train_ngram", "predict_ngram", "train_bpe", "tokenize"],
                        type=str, help="Select which activity to perform")
    parser.add_argument('--data', type=str,
                        help="Path to data corpus")
    parser.add_argument('--save', type=str,
                        help="Save model to specified path")
    parser.add_argument('--load', type=str,
                        help="Load model from specified path")
    parser.add_argument('--word', type=str,
                        help="Specify the first word(s) to predict")
    parser.add_argument('--nwords', type=int,
                        help="Select how many words to predict")
    parser.add_argument('--text', type=str,
                        help="Specify the string to be tokenized")
    parser.add_argument('--n', choices=[1, 2], type=int,
                        help="Choose which NGram model to use (1 or 2)")
    parser.add_argument('--d', action='store_true',
                        help="Set deterministic flag to True")

    # Instantiate a model and tokenizer for each class
    args = parser.parse_args()
    model = NGram(args.n)
    tokenizer = BPEAlg()

    if args.s == "train_ngram":
        if args.data:
            if os.path.isfile(args.data):
                with open(args.data, "r", encoding="utf-8") as file:
                    content = file.read()
                    model.train(content)
            else: print("Error: File not found.")
        else: print("Error: No data selected.")

    elif args.s == "predict_ngram":
        if args.load and args.nwords and args.word:
            predict = tuple(re.findall(r"\w+|[.,!?;]", args.word))
            with open(args.load, "rb") as file:
                model = pickle.load(file)
            if args.d:
                model.predict_next_word(predict, args.nwords, args.d)
            else: model.predict_next_word(predict, args.nwords)
        else: print("Error: File not loaded or word/number of words not specified.")

    elif args.s == "train_bpe":
        if args.data:
            if os.path.isfile(args.data):
                with open(args.data, "r", encoding="utf-8") as file:
                    content = file.read()
                    tokenizer.train(content)
            else: print("Error: File not found.")
        else: print("Error: No data selected.")

    elif args.s == "tokenize":
        if args.load and args.text:
            with open(args.load, "rb") as file:
                tokenizer = pickle.load(file)
            tokenizer.tokenize(args.text)
        else: print("Error: File not loaded or text not given.")

    if args.save:
        if args.s == "train_ngram":
            with open(args.save, "wb") as file:
                pickle.dump(model, file)
                print(f"Model saved to {args.save}")
        elif args.s == "train_bpe":
            with open(args.save, "wb") as file:
                pickle.dump(tokenizer, file)
                print(f"Model saved to {args.save}")
        else: print("Error: Must train data to save.")







