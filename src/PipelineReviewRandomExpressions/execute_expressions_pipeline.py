from src.utils.setup import setup
from src.PipelineReviewRandomExpressions.write_explanations import write_explanations_for_expressions
from src.PipelineReviewRandomExpressions.babble_labble_application import train_for_expressions
from src.PipelineReviewRandomExpressions.random_nouns_expressions_extraction import extract_nouns, extract_expressions
import os


filelist = ['data/train_examples.pkl', 'data/dev_examples.pkl', 'data/test_examples.pkl', 'data/train_labels.pkl'
                , 'data/dev_labels.pkl', 'data/test_labels.pkl', 'data/data.pkl', 'data/labels.pkl']
if not all([os.path.isfile(f) for f in filelist]):
    setup(199, 0, 199)

for i in range(1, 50):
    extract_nouns(i)
    extract_expressions(i)
    write_explanations_for_expressions(i)
    train_for_expressions(i, 0.3, 1, 3)
