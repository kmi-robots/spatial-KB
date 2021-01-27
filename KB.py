"""KB class"""
import json
import os
from onto import init_onto


class KnowledgeBase():

    def __init__(self, args):
        self.ontology = init_onto(args.IRI)
        # Load raw external KB data
        with open(os.path.join(args.path_to_data, 'relationships.json')) as ind:
            self.data = json.load(ind)
