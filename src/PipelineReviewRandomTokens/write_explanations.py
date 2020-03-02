import pickle
from babble import Explanation
from babble.utils import ExplanationIO2
import progressbar


def write_explanations_for_tokens(iteration_number):
    print("Writing explanations")

    DATA_FILE1 = 'data/tokens/tokens_train_list' + str(iteration_number) + '.pkl'
    DATA_FILE2 = 'data/explanations/my_explanations_tokens' + str(iteration_number) + '.tsv'

    with open(DATA_FILE1, 'rb') as f:
        tokens_list = pickle.load(f)

    index = 0
    explanations = []

    for selected_words in progressbar.progressbar(tokens_list):

        for word in selected_words:
            explanation = Explanation(
                name='LF_' + str(index),
                label=word[1],
                condition=create_condition_for_tokens(word[0]),
                word=word[0]
            )

            explanations.append(explanation)
            index = index + 1

    exp_io = ExplanationIO2()
    exp_io.write(explanations, DATA_FILE2)

    print("Done")


def create_condition_for_tokens(word):
    condition = 'the word ' + '"' + word + '" is in the sentence'
    return condition
