import pickle
import progressbar
from babble import Babbler
from babble import Explanation
from src.utils.utils import most_frequent, calculate_number_wrong_no_abstain

DATA_FILE1 = 'data/data.pkl'
DATA_FILE2 = 'data/labels.pkl'
DATA_FILE3 = 'data/explanations/my_explanations_expressions1.tsv'
DATA_FILE4 = 'data/expressions/correct_expressions_list1.pkl'



def test2():
    with open(DATA_FILE1, 'rb') as f:
        Cs = pickle.load(f)

    with open(DATA_FILE2, 'rb') as f:
        Ys = pickle.load(f)

    with open(DATA_FILE4, 'rb') as f:
        correct_expressions_list = pickle.load(f)

    index = 0
    explanations = []

    for expressions in progressbar.progressbar(correct_expressions_list):

        for expression in expressions:
            explanation = Explanation(
                name='LF_' + str(index),
                label=expression[1],
                condition=create_condition_for_expressions(expression[0]),
                word=expression[0]
            )

            explanations.append(explanation)
            index = index + 1


    babbler = Babbler(Cs, Ys)

    babbler.apply(explanations, split=0)

    Ls = []
    for split in [0, 1, 2]:
        L = babbler.get_label_matrix(split)
        Ls.append(L)

    babbler.commit()

    parses = babbler.get_parses()
    '''
    parse = parses[222]
    print(parse)
    '''
    L_train = Ls[0].toarray()
    L_test = Ls[2].toarray()

    predicted_training_labels = []
    predicted_test_labels = []

    for line in L_train:
        predicted_training_labels.append(most_frequent(line))

    for line in L_test:
        predicted_test_labels.append(most_frequent(line))

    len_wrong_train, training_accuracy = calculate_number_wrong_no_abstain(Ys[0], predicted_training_labels)

    len_wrong_test, test_accuracy = calculate_number_wrong_no_abstain(Ys[2], predicted_test_labels)

    print(predicted_training_labels)
    print(predicted_test_labels)
    print("Number of wrong in training set: " + str(len_wrong_train))
    print("Number of wrong in test set: " + str(len_wrong_test))
    print("Training Accuracy: " + str(training_accuracy))
    print("Test Accuracy: " + str(test_accuracy))


def create_condition_for_expressions(expression):
    condition = 'the phrase ' + '"' + expression + '" is in the sentence'
    return condition

test2()
