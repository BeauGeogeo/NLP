from __future__ import unicode_literals, print_function
import json
import plac
import random
from pathlib import Path

import spacy
from spacy.util import minibatch, compounding

from bs4 import BeautifulSoup


def clean_up_text4(string):
    string = BeautifulSoup(string, "html.parser").get_text()
    doc = nlp(string, disable=['parser', 'ner', 'textcat'])
    words = [token.text for token in doc if not token.is_stop]
    return " ".join(words)


nlp = spacy.load("en_core_web_lg")
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

with open("Data/JEOPARDY_QUESTIONS1.json") as f:
    TEXTS = json.loads(f.read())

TEXTS_OK = set()
examples = []
for e in TEXTS:
    e['question'] = clean_up_text4(e['question'])
    if 'historic' in e['category'].lower():
        del e['category']
        e['category'] = "HISTORY"
        examples.append(
            (e["question"], {'cats': {'HISTORY': e['category'] == 'HISTORY', 'SCIENCE': e['category'] == 'SCIENCE'}}))
    elif 'history' in e['category'].lower():
        del e['category']
        e['category'] = "HISTORY"
        examples.append(
            (e["question"], {'cats': {'HISTORY': e['category'] == 'HISTORY', 'SCIENCE': e['category'] == 'SCIENCE'}}))
    elif 'science' in e['category'].lower():
        del e['category']
        e['category'] = "SCIENCE"
        examples.append(
            (e["question"], {'cats': {'HISTORY': e['category'] == 'HISTORY', 'SCIENCE': e['category'] == 'SCIENCE'}}))
    elif 'scientific' in e['category'].lower():
        del e['category']
        e['category'] = "SCIENCE"
        examples.append(
            (e["question"], {'cats': {'HISTORY': e['category'] == 'HISTORY', 'SCIENCE': e['category'] == 'SCIENCE'}}))

print(f"Number of examples : {len(examples)}\n")

cpt1 = 0
cpt2 = 0

for e in examples:
    if e[1]['cats']['HISTORY']:
        cpt1 += 1
    else:
        cpt2 += 1

print(f"\nNumber of examples belonging to the HISTORY category : {cpt1}\n")
print(f"\nNumber of examples belonging to the SCIENCE category : {cpt2}\n")

train_data = examples
random.shuffle(train_data)

dev_texts = []
dev_cats = []

for e in train_data[5400:6600]:
    dev_texts += [e[0]]
    dev_cats += [e[1]['cats']]

test_texts = []
test_cats = []
for e in train_data[6600:]:
    test_texts += [e[0]]
    test_cats += [e[1]['cats']]

train_data = train_data[:5400]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_texts=("Number of texts to train from", "option", "t", int),
    n_iter=("Number of training iterations", "option", "n", int),
    init_tok2vec=("Pretrained tok2vec weights", "option", "t2v", Path)
)
def main(
        model=None,
        output_dir=None,
        n_iter=10,
        n_texts=7504,
        init_tok2vec=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat",
            config={
                "exclusive_classes": True,
                "architecture": "simple_cnn",
            }
        )
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe("textcat")

    # add label to text classifier
    textcat.add_label("HISTORY")
    textcat.add_label("SCIENCE")

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        if init_tok2vec is not None:
            with init_tok2vec.open("rb") as file_:
                textcat.model.tok2vec.from_bytes(file_.read())
        print("Training the model...")
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
        batch_sizes = compounding(4.0, 32.0, 1.001)
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                # print('texts : ', texts)
                # print('annotations : ', annotations)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            print('losses : ', losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print('losses dev_set : ', losses["textcat"])
            print("precision dev_set: ", scores)

            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print(
                "Loss : {0:.3f}\tPrecision : {1:.3f}\tRecall : {2:.3f}\tF-score : {3:.3f}".format(losses["textcat"],
                                                                                                  scores["textcat_p"],
                                                                                                  scores["textcat_r"],
                                                                                                  scores["textcat_f"],
                                                                                                  )
                # print a simple table
            )

    # test the trained model

    print("\nSome predictions on 'custom' sentences\n")

    test_text = "Science was much less advanced in antiquity"
    doc = nlp(test_text)
    print(test_text, doc.cats)

    test_text = "The earth is 4.5 billion years old"
    doc = nlp(test_text)
    print(test_text, doc.cats)

    test_text = "Edison was a great inventor."
    doc = nlp(test_text)
    print(test_text, doc.cats)

    test_text = "Edison was a great inventor. He discovered how to use electricity."
    doc = nlp(test_text)
    print(test_text, doc.cats)

    test_text = "Edison was a great inventor. Electricity was discovered by him."
    doc = nlp(test_text)
    print(test_text, doc.cats)

    test_text = "3 kings were ruling in Europe in 756 A.D."
    doc = nlp(test_text)
    print(test_text, doc.cats)

    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        print(test_text, doc2.cats)


def evaluate(tokenizer, textcat, texts, cats):  # demander explication sur cette partie...
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        print('doc : ', doc)
        gold = cats[i]  # vérifier si dico avec cats en clé (comme dans mon print de jeopardy_v3) ou pas
        print("gold : ", gold)
        for label, score in doc.cats.items():
            print('score : ', score)
            print('label : ', label)
            if label not in gold:
                continue
            if label == "SCIENCE":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 > gold[label]:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif gold[label] >= 0.5 > score:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


if __name__ == "__main__":
    plac.call(main)

# Some examples of predictions and the corresponding result

# Science was much less advanced in antiquity {'HISTORY': 0.0004234965890645981, 'SCIENCE': 0.9995765089988708}
# The earth is 4.5 billion years old {'HISTORY': 3.6052379073225893e-06, 'SCIENCE': 0.9999964237213135}
# Edison was a great inventor. {'HISTORY': 0.9756609797477722, 'SCIENCE': 0.024339064955711365}
# Edison was a great inventor. He discovered how to use electricity. {'HISTORY': 0.9354064464569092,
# 'SCIENCE': 0.06459353864192963}
# Edison was a great inventor. Electricity was discovered by him. {'HISTORY': 0.6420945525169373,
# 'SCIENCE': 0.35790547728538513}
# 3 kings were ruling in Europe in 756 A.D. {'HISTORY': 0.1120607852935791, 'SCIENCE': 0.8879392147064209}
