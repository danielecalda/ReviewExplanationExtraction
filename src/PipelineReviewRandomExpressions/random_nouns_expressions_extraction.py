import progressbar
import pickle
import random
import spacy
from src.utils.utils import check_adjective_noun, check_adjective_after_verb, check_adjective_adv_verb

DATA_FILE1 = 'data/train_examples.pkl'
DATA_FILE2 = 'data/train_labels.pkl'


def extract_nouns(iteration_number):
    DATA_FILE3 = 'data/expressions/nouns_list' + str(iteration_number) + '.pkl'

    with open(DATA_FILE1, 'rb') as f:
        examples = pickle.load(f)

    spacy_nlp = spacy.load('en_core_web_sm')

    try:
        with open(DATA_FILE3, 'rb') as f:
            nouns_list = pickle.load(f)
    except:
        nouns_list = []

    for example in progressbar.progressbar(examples):
        doc = spacy_nlp(example)

        nouns = [token.text for token in doc if not token.is_stop and token.is_alpha and token.tag_ in 'NN'
                 and token.ent_type_ not in ('PERSON', 'GPE', 'NORP', 'DATE', 'CARDINAL', 'LOC')]

        nouns_list.append(nouns)

    with open(DATA_FILE3, 'wb') as f:
        pickle.dump(nouns_list, f)


def extract_expressions(iteration_number):
    DATA_FILE3 = 'data/expressions/nouns_list' + str(iteration_number) + '.pkl'
    DATA_FILE4 = 'data/expressions/expressions_train_list' + str(iteration_number) + '.pkl'
    DATA_FILE5 = 'data/expressions/correct_expressions_list' + str(iteration_number - 1) + '.pkl'
    DATA_FILE6 = 'data/expressions/wrong_expressions_list' + str(iteration_number - 1) + '.pkl'

    with open(DATA_FILE3, 'rb') as f:
        nouns_list = pickle.load(f)

    with open(DATA_FILE1, 'rb') as f:
        train_examples = pickle.load(f)

    with open(DATA_FILE2, 'rb') as f:
        train_labels = pickle.load(f)

    try:
        with open(DATA_FILE5, 'rb') as f:
            correct_expressions_list = pickle.load(f)
    except:
        correct_expressions_list = [[] for i in range(len(train_examples))]
    try:
        with open(DATA_FILE6, 'rb') as f:
            wrong_expressions_list = pickle.load(f)
    except:
        wrong_expressions_list = [[] for i in range(len(train_examples))]

    expressions_train_list = []

    print(len(train_examples))
    print(len(nouns_list))

    for example, label, nouns, correct_expressions, wrong_expressions in progressbar.progressbar(zip(train_examples,
                                    train_labels, nouns_list, correct_expressions_list, wrong_expressions_list)):

        expressions = []

        if len(nouns) >= 3:
            indexes = random.sample(range(len(nouns)), 3)
            for index in indexes:
                expression = None
                adjective = check_adjective_adv_verb(example, nouns[index])
                if adjective is not None:
                    expression = (nouns[index] + ' ' + adjective, label)
                else:
                    adjective = check_adjective_after_verb(example, nouns[index])
                    if adjective is not None:
                        expression = (nouns[index] + ' ' + adjective, label)
                    else:
                        adjective = check_adjective_noun(example, nouns[index])
                        if adjective is not None:
                            expression = (adjective + ' ' + nouns[index], label)
                if expression is not None and expression not in correct_expressions and expression not in wrong_expressions:
                    expressions.append(expression)
        expressions_train_list.append(expressions)

    with open(DATA_FILE4, 'wb') as f:
        pickle.dump(expressions_train_list, f)
