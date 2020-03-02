import pickle
from src.utils.utils import most_frequent, calculate_number_wrong_no_abstain, \
    create_tokens_from_choiced_explanations, high_coverage_elements, high_correct_elements, intersection,\
    average
from metal import LabelModel

DATA_FILE1 = 'data/data.pkl'
DATA_FILE2 = 'data/labels.pkl'


def analyze_for_tokens(ls, parses, iteration_number, coverage_treshold, correct_treshold, wrong_treshold, modality='most'):

    DATA_FILE5 = 'data/results/predicted_training_labels'  + str(iteration_number) + '.pkl'
    DATA_FILE6 = 'data/tokens/correct_tokens_list' + str(iteration_number) + '.pkl'
    DATA_FILE11 = 'data/tokens/correct_tokens_list' + str(iteration_number - 1) + '.pkl'
    DATA_FILE7 = 'data/results/summary.txt'
    DATA_FILE8 = 'data/results/predicted_test_labels' + str(iteration_number) + '.pkl'
    DATA_FILE9 = 'data/tokens/tokens_train_list' + str(iteration_number) + '.pkl'
    DATA_FILE10 = 'data/tokens/wrong_tokens_list' + str(iteration_number) + '.pkl'
    DATA_FILE12 = 'data/tokens/wrong_tokens_list' + str(iteration_number -1) + '.pkl'

    with open(DATA_FILE2, 'rb') as f:
        Ys = pickle.load(f)

    with open(DATA_FILE9, 'rb') as f:
        tokens_train_list = pickle.load(f)

    L_train = ls[0].toarray()
    L_test = ls[2].toarray()

    predicted_training_labels = []
    predicted_test_labels = []

    if modality == 'most':
        for line in L_train:
            predicted_training_labels.append(most_frequent(line))

        for line in L_test:
            predicted_test_labels.append(most_frequent(line))
    elif modality == 'avg':
        for line in L_train:
            predicted_training_labels.append(average(line))

        for line in L_test:
            predicted_test_labels.append(average(line))
    elif modality == 'label_agg':
        label_aggregator = LabelModel(5)
        label_aggregator.train(ls[0], n_epochs=100, lr=0.003)
        label_aggregator.score(ls[1], Ys[1])
        predicted_training_labels = label_aggregator.predict(ls[0])
        predicted_test_labels = label_aggregator.predict(ls[2])


    with open(DATA_FILE5, 'wb') as f:
        pickle.dump(predicted_training_labels, f)

    with open(DATA_FILE8, 'wb') as f:
        pickle.dump(predicted_test_labels, f)

    len_wrong_train, training_accuracy = calculate_number_wrong_no_abstain(Ys[0], predicted_training_labels)

    len_wrong_test, test_accuracy = calculate_number_wrong_no_abstain(Ys[2], predicted_test_labels)

    with open(DATA_FILE7, 'a') as f:
        f.write("Iteration number: " + str(iteration_number))
        f.write('\n')
        f.write("Number of wrong in training set: " + str(len_wrong_train))
        f.write('\n')
        f.write("Number of wrong in test set: " + str(len_wrong_test))
        f.write('\n')
        f.write("Training Accuracy: " + str(training_accuracy))
        f.write('\n')
        f.write("Test Accuracy: " + str(test_accuracy))
        f.write('\n')
        f.write('\n')

    L_train_transpose = L_train.T

    over_percentage = high_coverage_elements(L_train_transpose, coverage_treshold)
    correct_elements, wrong_elements = high_correct_elements(L_train_transpose, Ys, correct_treshold, wrong_treshold)

    intersect = intersection(over_percentage, correct_elements)

    new_explanations = []
    for index in intersect:
        explanation = parses[index].explanation
        new_explanations.append(explanation)

    wrong_explanations = []
    for index in wrong_elements:
        wrong_explanation = parses[index].explanation
        wrong_explanations.append(wrong_explanation)

    correct_token_from_explanations = create_tokens_from_choiced_explanations(new_explanations)
    wrong_token_from_explanations = create_tokens_from_choiced_explanations(wrong_explanations)

    if iteration_number % 5 == 0 or iteration_number == 1:
        correct_tokens_list = [[] for i in range(len(tokens_train_list))]
        wrong_tokens_list = [[] for i in range(len(tokens_train_list))]
    else:
        with open(DATA_FILE11, 'rb') as f:
            correct_tokens_list = pickle.load(f)
        with open(DATA_FILE12, 'rb') as f:
            wrong_tokens_list = pickle.load(f)

    for i, tokens_list in enumerate(tokens_train_list):
        correct_tokens = correct_tokens_list[i]
        wrong_tokens = wrong_tokens_list[i]
        for token in tokens_list:
            if token in correct_token_from_explanations and token not in correct_tokens:
                correct_tokens_list[i].append(token)
            if token in wrong_token_from_explanations and token not in wrong_tokens:
                wrong_tokens_list[i].append(token)

    with open(DATA_FILE6, 'wb') as f:
        pickle.dump(correct_tokens_list, f)

    with open(DATA_FILE10, 'wb') as f:
        pickle.dump(wrong_tokens_list, f)
