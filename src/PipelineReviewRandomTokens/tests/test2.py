import pickle
import progressbar
from babble import Babbler
from babble import Explanation
from src.utils.utils import most_frequent, calculate_number_wrong_no_abstain

DATA_FILE1 = '../data/data.pkl'
DATA_FILE2 = '../data/labels.pkl'

def test2(i):
    DATA_FILE4 = '../data/tokens/correct_tokens_list' + str(i) + '.pkl'

    with open(DATA_FILE4, 'rb') as f:
        correct_tokens_list = pickle.load(f)

    print(correct_tokens_list[187])


for i in range(1, 39):
    test2(i)
