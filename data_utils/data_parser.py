import os
import json
import numpy as np

from collections import namedtuple
from ipdb import set_trace
ListData = namedtuple('ListData', ['id', 'label', 'path'])


class DatasetBase(object):
    """
    To read json data and construct a list containing video sample `ids`,
    `label` and `path`
    """
    def __init__(self, json_path_input, json_path_labels, data_root,
                 extension, is_test=False):
        self.json_path_input = json_path_input
        self.json_path_labels = json_path_labels
        self.data_root = data_root
        self.extension = extension
        self.is_test = is_test

        # preparing data and class dictionary
        self.classes = self.read_json_labels()  # dict ['Monving something up' = 76]
        self.json_data = self.read_json_input()

    def read_json_input(self):
        json_data = []
        train_data1 = []

        with open(self.json_path_input, 'r') as jsonfile:
            json_reader = json.load(jsonfile)
            for elem in json_reader:
                # add a dummy label for all test samples
                item = ListData(elem['id'],
                                "Holding something",
                                os.path.join(self.data_root,
                                             elem['id'] + self.extension)
                                )
                json_data.append(item)
        return json_data

    def read_json_labels(self):

        with open(self.json_path_labels, 'r') as jsonfile:
            # print(jsonfile)
            json_reader = json.load(jsonfile)

        return json_reader

    def get_two_way_dict(self, classes):
        classes_dict = {}
        for i, item in enumerate(classes):
            classes_dict[item] = i
            classes_dict[i] = item
        return classes_dict

    def clean_template(self, template):
        """ Replaces instances of `[something]` --> `something`"""
        template = template.replace("[", "")
        template = template.replace("]", "")
        return template


class WebmDataset(DatasetBase):
    def __init__(self, json_path_input, json_path_labels, data_root,
                 is_test=False):
        EXTENSION = ".webm"
        super().__init__(json_path_input, json_path_labels, data_root,
                         EXTENSION, is_test)

