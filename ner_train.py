import random
import re
import json
from pathlib import Path

import spacy


class NERTrain:

    def __init__(self, labels, model=None):
        self.__load_model(model, labels)
        self.training_data = None

    def __load_model(self, model, labels):
        if model is not None:
            nlp = spacy.load(model)
        else:
            nlp = spacy.blank('en')
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner)
        else:
            ner = nlp.get_pipe('ner')
        if model is None:
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.entity.create_optimizer()
        for label in labels:
            ner.add_label(label)
        self.__dict__['nlp'] = nlp
        self.__dict__['optimizer'] = optimizer
        self.__dict__['ner'] = ner

    def __json_to_training_object(self, item):
        sentence = item.get('sentence')
        phrases = item.get('phrases')
        named_ents = item.get('entities')
        entities = []
        for idx, value in enumerate(phrases):
            result = [(m.start(), m.end(), named_ents[idx]) for m in re.finditer(value, sentence)]
            if result:
                entities.append(result[0])
        return entities

    def __build_training_object(self, item):
        entity_info = self.__json_to_training_object(item)
        return item.get('sentence'), {'entities': entity_info}

    def load_training_json(self, json_file):
        training_data = []
        with open(json_file, 'r') as f:
            data = json.load(f)
        for item in data:
            training_data.append(self.__build_training_object(item))
        self.training_data = training_data

    def train_model(self, iterations=10, drop=0.35):
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'ner']
        with self.nlp.disable_pipes(*other_pipes):
            for it in range(iterations):
                random.shuffle(self.training_data)
                losses = {}
                for text, annotations in self.training_data:
                    self.nlp.update([text], [annotations], sgd=self.optimizer, drop=drop,
                                    losses=losses)
                print(losses)

    def save_model(self, output_dir, model_name):
        self.nlp.meta['name'] = model_name
        self.nlp.to_disk(Path(output_dir))
        print("Saved model: {}, to directory: {}".format(model_name, Path(output_dir)))

    def predict_entities(self, text):
        doc = self.nlp(text)
        return [(ent.label_, ent.text) for ent in doc.ents]


if __name__ == '__main__':
    n = NERTrain(["ANIMAL"])
    n.load_training_json("train.json")
    n.train_model()
    #n.save_model("/example-model", "play")
    print(n.predict_entities("Do you like horses?"))