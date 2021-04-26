"""Command Line Interface"""

import sys
import argparse
from KB import KnowledgeBase
from object_reasoner import ObjectReasoner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data', help='Base path to raw spatial data', default='./data')
    parser.add_argument('--dbname', help='Name for PostGRE SQL spatial database', default='gis_database')
    args = parser.parse_args()
    KB = KnowledgeBase(args)
    reasoner = ObjectReasoner(KB, args)
    reasoner.db_session()
    return 0


if __name__ == '__main__':
    sys.exit(main())
