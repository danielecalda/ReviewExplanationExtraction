import pickle
from babble import Explanation
import progressbar
from babble import Babbler


DATA_FILE1 = 'data/data.pkl'
DATA_FILE2 = 'data/labels.pkl'
DATA_FILE3 = 'data/tokens/correct_tokens_list12.pkl'
DATA_FILE4 = 'data/tokens/wrong_tokens_list2.pkl'
DATA_FILE5 = 'data/tokens/tokens_train_list2.pkl'
DATA_FILE6 = 'data/explanations/'


with open(DATA_FILE1, 'rb') as f:
    Cs = pickle.load(f)

with open(DATA_FILE2, 'rb') as f:
    Ys = pickle.load(f)

with open(DATA_FILE3, 'rb') as f:
    correct_tokens_list = pickle.load(f)

with open(DATA_FILE4, 'rb') as f:
    wrong_tokens_list = pickle.load(f)

