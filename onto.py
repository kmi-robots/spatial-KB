"""Ontology creation/manipulation
"""
from owlready2 import *
import types

def init_onto(IRI) -> Ontology:
    # Import base spatial ontology from IRI
    # Load geosparql from local first (fixed DublinCore link)
    geosparql = get_ontology('./data/ont/geosparql_vocab_all.rdf').load()
    onto = get_ontology(IRI).load() # this owl extens geosparql so it is based on the local fixed version now
    return onto
