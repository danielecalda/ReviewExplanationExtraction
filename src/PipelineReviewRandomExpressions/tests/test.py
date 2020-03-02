import pickle
from babble import Explanation
import progressbar
from babble import Babbler
from src.utils.utils import most_frequent
from src.utils.utils import calculate_number_wrong
from src.utils.utils import percentage

DATA_FILE1 = 'data/data.pkl'
DATA_FILE2 = 'data/labels.pkl'
DATA_FILE7 = 'data/results/summary.txt'


def apply_results():

    best_iteration = get_best_iteration()

    print(best_iteration)

    DATA_FILE3 = 'data/tokens/correct_expressions_list' + str(best_iteration[1]) + '.pkl'
    DATA_FILE4 = 'data/tokens/wrong_expressions_list' + str(best_iteration[1]) + '.pkl'
    DATA_FILE5 = 'data/tokens/expressions_train_list' + str(best_iteration[1]) + '.pkl'

    with open(DATA_FILE1, 'rb') as f:
        Cs = pickle.load(f)

    with open(DATA_FILE2, 'rb') as f:
        Ys = pickle.load(f)

    with open(DATA_FILE3, 'rb') as f:
        correct_tokens_list = pickle.load(f)

    with open(DATA_FILE4, 'rb') as f:
        wrong_tokens_list = pickle.load(f)

    with open(DATA_FILE5, 'rb') as f:
        tokens_train_list = pickle.load(f)

    tokens = []
    for correct_tokens in correct_tokens_list:
        for token in correct_tokens:
            if token not in tokens:
                tokens.append(token)

    tokens2 = []
    for tokens_train in tokens_train_list:
        for token in tokens_train:
            if token not in tokens2:
                tokens2.append(token)
    index = 0
    explanations = []

    for word in progressbar.progressbar(tokens2):
        explanation = Explanation(
            name='LF_' + str(index),
            label=word[1],
            condition=create_condition(word[0]),
            word=word[0]
        )

        explanations.append(explanation)
        index = index + 1

    babbler = Babbler(Cs, Ys)

    babbler.apply(explanations, split=0)

    Ls = []
    for split in [0, 1, 2]:
        L = babbler.get_label_matrix(split)
        Ls.append(L)

    L_train = Ls[0].toarray()
    L_test = Ls[2].toarray()

    predicted_training_labels = []
    predicted_test_labels = []

    for line in L_train:
        predicted_training_labels.append(most_frequent(line))
    count = 0
    for label in predicted_training_labels:
        if label != 0:
            count += 1
    print(count)

    for line in L_test:
        predicted_test_labels.append(most_frequent(line))

    len_wrong_train = calculate_number_wrong(Ys[0], predicted_training_labels)

    len_wrong_test = calculate_number_wrong(Ys[2], predicted_test_labels)

    training_accuracy = percentage(len(Ys[0]) - len_wrong_train, len(Ys[0]))

    test_accuracy = percentage(len(Ys[2]) - len_wrong_test, len(Ys[2]))

    print("Number of wrong in training set: " + str(len_wrong_train))
    print("Number of wrong in test set: " + str(len_wrong_test))
    print("Training Accuracy: " + str(training_accuracy))
    print("Test Accuracy: " + str(test_accuracy))


def create_condition(word):
    condition = 'the word ' + '"' + word + '" is in the sentence'
    return condition

def get_best_iteration():
    summary = open(DATA_FILE7, "r")
    text = summary.readlines()
    best = (0,0)
    iteration = 0
    for line in text:
        if 'Test' in line:
            iteration += 1
            test_accuracy = float(line[15:19])
            if test_accuracy > best[0]:
                best = (test_accuracy, iteration)
    return best

#print(correct_tokens_list)
apply_results()