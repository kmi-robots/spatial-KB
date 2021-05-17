"""
class to handle interaction with spatial DB
"""

from PostGIS import *

class SpatialDB():

    def __init__(self, KBobj, args):
        self.KB = KBobj
        self.db_user = os.environ['USER']
        self.dbname = args.dbname

    def db_session(self): # Things we do offline for all images at once
        # Open connection to PostGRE SQL DB
        self.connection, self.cursor = connect_DB(self.db_user,self.dbname)

        if self.connection is not None:
            # Load knowledge base
            #self.KB = self.KB.load_data(self)
            # Create 3D spatial abstractions for objects in map and update spatial table
            create_boxes(self) # done once for all objects
            # Commit all changes to DB
            self.connection.commit()
            # Close connection, to avoid db issues
            disconnect_DB(self.connection, self.cursor)