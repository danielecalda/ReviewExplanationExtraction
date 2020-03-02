import spacy

spacy_nlp = spacy.load('en_core_web_sm')

doc = spacy_nlp("there is a great atmosphere in the city but is not good")

pos_sentence = ''
for token in doc:
    pos_sentence = pos_sentence + ' ' + token.pos_

print(doc)
print(pos_sentence)



