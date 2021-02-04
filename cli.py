"""Command Line Interface"""

import sys
import argparse
from KB import KnowledgeBase
from object_reasoner import ObjectReasoner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data', help="Base path to raw spatial data", default='./data')
    parser.add_argument('--IRI', help="Reference ontology IRI", default='./data/ont/spatial-onto.owl')
    parser.add_argument('--dbname', help="Name for PostGRE SQL spatial database", default='VG_spatial')
    args = parser.parse_args()
    KB = KnowledgeBase(args)
    reasoner = ObjectReasoner(KB,args)
    reasoner.db_session()
    return 0

if __name__ == "__main__":
    sys.exit(main())