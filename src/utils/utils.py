from collections import Counter
import re
import spacy


def most_frequent(line):
    filtered_list = list(filter(lambda a: a != 0, line))
    if len(filtered_list) > 0:
        occurence_count = Counter(filtered_list)
        return occurence_count.most_common(1)[0][0]
    else:
        return 0


def average(line):
    filtered_list = list(filter(lambda a: a != 0, line))
    if len(filtered_list) > 0:
        return sum(filtered_list)/len(filtered_list)
    else:
        return 0


def percentage(part, whole):
    return 100 * float(part)/float(whole)


def create_tokens_from_choiced_explanations(explanations):
    tokens_from_explanations = []
    for explanation in explanations:
        word = explanation.word
        label = explanation.label
        new = (word, label)
        tokens_from_explanations.append(new)
    return tokens_from_explanations


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def calculate_number_wrong(real_labels, predicted_labels):
    len_wrong = 0
    for real, predicted in zip(real_labels, predicted_labels):
        if real != predicted:
            len_wrong += 1
    return len_wrong


def calculate_number_wrong_no_abstain(real_labels, predicted_labels):
    len_wrong = 0
    no_abstain = 0
    for real, predicted in zip(real_labels, predicted_labels):
        if predicted != 0:
            no_abstain += 1
            if real != predicted and predicted != 0:
                len_wrong += 1
    if no_abstain > 0:
        accuracy = percentage(no_abstain - len_wrong, no_abstain)
    else:
        accuracy = percentage(len(predicted_labels) - len_wrong, len(predicted_labels))
    return len_wrong, accuracy


def average_coverage_elements(l_train):
    coverages = []
    for i, line in enumerate(l_train):
        count = 0
        for element in line:
            if element != 0:
                count += 1
        coverage = percentage(count, len(line))
        coverages.append(coverage)
    return sum(coverages)/len(coverages)


def high_coverage_elements(l_train, treshold):
    over_percentage = []
    for i, line in enumerate(l_train):
        count = 0
        for element in line:
            if element != 0:
                count += 1
        coverage = percentage(count, len(line))
        if coverage > treshold:
            over_percentage.append(i)
    print('number of over percentage is: ' + str(len(over_percentage)))
    return over_percentage


def high_correct_elements(l_train, ys, correct_treshold, wrong_treshold):
    correct_elements = []
    wrong_elements = []
    for i, line in enumerate(l_train):
        abstain = 0
        correct = 0
        wrong = 0
        for element, label in zip(line, ys[0]):
            if element == 0:
                abstain += 1
            else:
                if element == label:
                    correct += 1
                else:
                    wrong += 1
        if correct > correct_treshold*wrong:
            correct_elements.append(i)
        elif wrong > wrong_treshold*correct:
            wrong_elements.append(i)
    print('number of correct elements is: ' + str(len(correct_elements)))
    return correct_elements, wrong_elements


def high_correct_elements2(l_train):
    correct_elements = []
    wrong_elements = []
    for i, line in enumerate(l_train):
        counter = Counter(line)
        values = list(counter.values())
        if len(values) == 1:
            correct_elements.append(i)
            # print('abstain: ' + str(abstain) + ' and correct: ' + str(correct) + ' and wrong: ' + str(wrong))
    print('number of correct elements is: ' + str(len(correct_elements)))
    return correct_elements, wrong_elements


def get_words_after(quantity,sentence,entity):
    sentence = re.sub(r'[^\w\s]','',sentence)
    words = sentence.split()
    if entity in words:
        index = words.index(entity) +1
        after = index + min(index, quantity)
        return ' '.join(map(str, words[index:after]))


def get_words_before(quantity,sentence,entity):
    sentence = re.sub(r'[^\w\s]','',sentence)
    words = sentence.split()
    if entity in words:
        index = words.index(entity)
        before = index - min(index, quantity)
        return ' '.join(map(str, words[before:index]))


def check_adjective_noun(sentence, entity):
    words = get_words_before(1,sentence,entity)
    if words == None or len(words.split(" ")) == 0:
        return None
    else:
        spacy_nlp = spacy.load('en_core_web_sm')
        doc = spacy_nlp(words)
        for token in doc:
            if token.pos_ == 'ADJ':
                return words
                break


def check_adjective_after_verb(sentence,entity):
    words = get_words_after(2,sentence,entity)
    if words == None or len(words.split(" ")) < 2:
        return None
    else:
        spacy_nlp = spacy.load('en_core_web_sm')
        doc = spacy_nlp(words)
        if doc[0].pos_ == 'VERB' and doc[1].pos_ == 'ADJ':
            return words


def check_adjective_adv_verb(sentence,entity):
    words = get_words_after(3,sentence,entity)
    if words == None or len(words.split(" ")) < 3:
        return None
    else:
        spacy_nlp = spacy.load('en_core_web_sm')
        doc = spacy_nlp(words)
        if doc[0].pos_ == 'VERB' and doc[0].pos_ == 'ADV' and doc[1].pos_ == 'ADJ':
            return words