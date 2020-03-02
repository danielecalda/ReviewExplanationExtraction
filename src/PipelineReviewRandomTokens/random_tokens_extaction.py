import pickle
import random
import spacy
import progressbar


DATA_FILE1 = 'data/train_examples.pkl'
DATA_FILE2 = 'data/train_labels.pkl'


def extract_token(iteration_number):
    print("Extracting tokens")

    DATA_FILE3 = 'data/tokens/correct_tokens_list' + str(iteration_number - 1) + '.pkl'
    DATA_FILE4 = 'data/tokens/tokens_train_list' + str(iteration_number) + '.pkl'
    DATA_FILE5 = 'data/tokens/wrong_tokens_list' + str(iteration_number - 1) + '.pkl'

    with open(DATA_FILE1, 'rb') as f:
        examples = pickle.load(f)
    with open(DATA_FILE2, 'rb') as f:
        labels = pickle.load(f)
    try:
        with open(DATA_FILE3, 'rb') as f:
            correct_tokens_list = pickle.load(f)
    except:
        correct_tokens_list = [[] for i in range(len(examples))]
    try:
        with open(DATA_FILE5, 'rb') as f:
            wrong_tokens_list = pickle.load(f)
    except:
        wrong_tokens_list = [[] for i in range(len(examples))]

    tokens_train_list = []

    spacy_nlp = spacy.load('en_core_web_sm')

    print(len(examples))
    print(len(labels))
    print(len(correct_tokens_list))
    print(len(wrong_tokens_list))

    for example, label, correct_tokens, wrong_tokens in progressbar.progressbar(zip(examples, labels, correct_tokens_list, wrong_tokens_list)):
        doc = spacy_nlp(example)
        tokenized_sentence = [token.text for token in doc if not token.is_stop and token.is_alpha and token.pos_ not in ('SYM')]

        selected_words = select_random_words(tokenized_sentence, label, correct_tokens, wrong_tokens)
        tokens_train_list.append(selected_words)

    print('Correct tokens: ' + str(correct_tokens_list[438]))
    print('Wrong tokens: ' + str(wrong_tokens_list[438]))
    print('Choiced tokens: ' + str(tokens_train_list[438]))

    with open(DATA_FILE4, 'wb') as f:
        pickle.dump(tokens_train_list, f)

    print("Done")


def select_random_words(tokens, label, correct_tokens, wrong_tokens):
    selected_words = []
    wrong_indexes = []
    choiced_indexes = []
    if len(correct_tokens) < 5:
        while len(selected_words) < 3 and len(choiced_indexes) + len(wrong_indexes) < len(tokens):
            index = random.sample(range(len(tokens)), 1)[0]
            if index not in wrong_indexes and index not in choiced_indexes:
                new = (tokens[index], label)
                if new not in correct_tokens and new not in wrong_tokens:
                    selected_words.append(new)
                    choiced_indexes.append(index)
                else:
                    wrong_indexes.append(index)
        return selected_words + correct_tokens
    else:
        return correct_tokens

