"""
Reasoner class
"""
import os
from PostGIS import *
from halfspace import *

class ObjectReasoner():
    def __init__(self, KBobj, args):
        self.KB = KBobj
        self.db_user = os.environ['USER']
        self.dbname = args.dbname

    def db_session(self):
        # Open connection to PostGRE SQL DB
        self.connection, self.cursor = connect_DB(self.db_user,self.dbname)

        if self.connection is not None:
            # Load knowledge base
            self.KB = self.KB.load_data(self)

            # Query semantic map for QSR
            self.localQSR = self.query_map()

            # Validate and correct based on KB relations

            # Update local QSR DB with new predictions

            # Commit all changes to DB
            self.connection.commit()
            # Close connection, to avoid db issues
            disconnect_DB(self.connection, self.cursor)

    def query_map(self):
        #extract all min 3D booxes for all polyhedrons in spatial DB
        extracted_bboxes = query_all_bboxes(self)

        # Extrude 3D bboxes based on halfspace projection model
        extracted_bboxes = compute_hs_projections(extracted_bboxes)

        # Create QSR graph
        # For each 3D object in map (optional: meeting some selection criteria)
        # Find nearby 3D objects and their min 3D bounding box
        localQSR = extract_QSRs(self,extracted_bboxes)

        return localQSR