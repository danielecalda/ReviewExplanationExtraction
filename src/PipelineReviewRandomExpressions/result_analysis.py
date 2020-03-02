import pickle
from src.utils.utils import most_frequent, calculate_number_wrong_no_abstain, \
    create_tokens_from_choiced_explanations, high_coverage_elements, high_correct_elements, intersection,\
    average
from metal import LabelModel

DATA_FILE1 = 'data/data.pkl'
DATA_FILE2 = 'data/labels.pkl'


def analyze_for_expressions(ls, parses, iteration_number, coverage_treshold, correct_treshold, wrong_treshold, modality='most'):

    DATA_FILE5 = 'data/results/predicted_training_labels'  + str(iteration_number) + '.pkl'
    DATA_FILE6 = 'data/expressions/correct_expressions_list' + str(iteration_number) + '.pkl'
    DATA_FILE7 = 'data/results/summary.txt'
    DATA_FILE8 = 'data/results/predicted_test_labels' + str(iteration_number) + '.pkl'
    DATA_FILE9 = 'data/expressions/expressions_train_list' + str(iteration_number) + '.pkl'
    DATA_FILE10 = 'data/expressions/wrong_expressions_list.pkl'

    with open(DATA_FILE2, 'rb') as f:
        Ys = pickle.load(f)

    with open(DATA_FILE9, 'rb') as f:
        expressions_train_list = pickle.load(f)

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
        label_aggregator.train(ls[0], n_epochs=50, lr=0.003)
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

    print(new_explanations)

    expressions_from_explanations = create_tokens_from_choiced_explanations(new_explanations)
    wrong_expressions_from_explanations = create_tokens_from_choiced_explanations(wrong_explanations)

    correct_expressions_list = []
    wrong_expressions_list = []

    for expressions_list in expressions_train_list:
        correct_expressions = []
        wrong_expressions = []
        for expression in expressions_list:
            if expression in expressions_from_explanations:
                correct_expressions.append(expression)
            if expression in wrong_expressions_from_explanations:
                wrong_expressions.append(expression)
        correct_expressions_list.append(correct_expressions)
        wrong_expressions_list.append(wrong_expressions)

    with open(DATA_FILE6, 'wb') as f:
        pickle.dump(correct_expressions_list, f)

    with open(DATA_FILE10, 'wb') as f:
        pickle.dump(wrong_expressions_list, f)
