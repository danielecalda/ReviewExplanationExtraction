import spacy

spacy_nlp = spacy.load('en_core_web_sm')

doc = spacy_nlp("Total bill for this horrible service? Over $8Gs. These crooks actually had the nerve to charge us $69 for 3 pills. "
                "I checked online the pills can be had for 19 cents EACH! Avoid Hospital ERs at all costs.")

pos_sentence = ''
for token in doc:
    pos_sentence = pos_sentence + ' ' + token.pos_

print(doc)
print(pos_sentence)



