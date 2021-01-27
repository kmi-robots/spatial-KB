"""Ontology creation/manipulation
"""
from owlready2 import *
import types

def init_onto(IRI) -> Ontology:
    # Import base spatial ontology from IRI
    onto = get_ontology(IRI).load()
    with onto:
        for cl in onto.classes(): print(cl)
    return onto
