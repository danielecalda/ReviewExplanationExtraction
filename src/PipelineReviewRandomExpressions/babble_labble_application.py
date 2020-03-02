import pickle
from babble.utils import ExplanationIO2
from babble import Babbler
from src.PipelineReviewRandomExpressions.result_analysis import analyze_for_expressions

DATA_FILE1 = 'data/data.pkl'
DATA_FILE2 = 'data/labels.pkl'


def train_for_expressions(iteration_number, coverage_treshold, correct_treshold, wrong_treshold, modality='most'):
    DATA_FILE3 = 'data/explanations/my_explanations_expressions' + str(iteration_number) + '.tsv'
    DATA_FILE4 = 'data/Ls/Ls' + str(iteration_number) + '.pkl'

    print("Start training")

    with open(DATA_FILE1, 'rb') as f:
        Cs = pickle.load(f)

    with open(DATA_FILE2, 'rb') as f:
        Ys = pickle.load(f)

    exp_io = ExplanationIO2()
    explanations = exp_io.read(DATA_FILE3)

    babbler = Babbler(Cs, Ys)

    babbler.apply(explanations, split=0)

    parses = babbler.get_parses(translate=False)

    Ls = []
    for split in [0, 1, 2]:
        L = babbler.get_label_matrix(split)
        Ls.append(L)

    with open(DATA_FILE4, 'wb') as f:
        pickle.dump(Ls, f)

    analyze_for_expressions(Ls, parses, iteration_number, coverage_treshold, correct_treshold, wrong_treshold, modality)

    print("Done")
