import pickle
from babble import Explanation
from babble.utils import ExplanationIO2
import progressbar


def write_explanations_for_expressions(iteration_number):
    print("Writing explanations")

    DATA_FILE1 = 'data/expressions/expressions_train_list' + str(iteration_number) + '.pkl'
    DATA_FILE2 = 'data/explanations/my_explanations_expressions' + str(iteration_number) + '.tsv'

    with open(DATA_FILE1, 'rb') as f:
        expressions_list = pickle.load(f)

    index = 0
    explanations = []

    for expressions in progressbar.progressbar(expressions_list):

        for expression in expressions:
            explanation = Explanation(
                name='LF_' + str(index),
                label=expression[1],
                condition=create_condition_for_expressions(expression[0]),
                word=expression[0]
            )

            explanations.append(explanation)
            index = index + 1

    exp_io = ExplanationIO2()
    exp_io.write(explanations, DATA_FILE2)

    print("Done")


def create_condition_for_expressions(expression):
    condition = 'the phrase ' + '"' + expression + '" is in the sentence'
    return condition