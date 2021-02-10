"""
Reasoner class
"""
import os
import networkx as nx
from PostGIS import *
from halfspace import *
from utils.graphs import plot_graph

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
            self.localQSR,self.globalQSR = self.query_map()

            # TODO Validate and correct based on KB relations

            # TODO Update spatial DB with new predictions

            # Commit all changes to DB
            self.connection.commit()
            # Close connection, to avoid db issues
            disconnect_DB(self.connection, self.cursor)

    def query_map(self):
        #extract all min 3D booxes for all polyhedrons in spatial DB
        extracted_bboxes = query_all_bboxes(self)

        # Extrude 3D bboxes based on halfspace projection model
        extracted_bboxes = compute_hs_projections(extracted_bboxes)

        #Initialise QSR graph
        globalQSR = nx.MultiDiGraph()
        globalQSR.add_nodes_from(extracted_bboxes.keys())  # add one node per object_id
        #for v in globalQSR.nodes(): globalQSR.nodes[v]["name"] = "object_"+v
        plot_graph(globalQSR)
        # For each 3D object in map (optional: meeting some selection criteria)
        # Find nearby 3D objects and their min 3D bounding box
        for object_id in extracted_bboxes:
            query_results = query_map_neighbours(self,extracted_bboxes,object_id)
            globalQSR = self.update_QSR_graph(query_results,object_id,globalQSR)
            # local graph including only neighbours
            localQSR = globalQSR.subgraph([1,2,3])
        plot_graph(globalQSR)
        return localQSR,globalQSR

    def update_QSR_graph(self,results,i,globalg):
        # Express QSRs as a graph and set rules for inverse relations
        for n, ops in results:
            intersects=False
            if not globalg.has_edge(i, n):  # if object and neighbour not connected already

                # avoiding duplicates and reducing computational cost
                vol_intersection = ops[3]
                if vol_intersection>0.:
                    volg1 = ops[1]
                    volg2 = ops[2]
                    intersects = True

                    if vol_intersection == volg1: #g1 is within g2 and g2 contains g1
                        globalg.add_edge(i, n, QSR="Intersects", ext="IsWithin")
                        globalg.add_edge(n, i, QSR="Intersects", ext="Contains")
                    elif vol_intersection == volg2: #g2 is within g1 and g1 contains g2
                        globalg.add_edge(i, n, QSR="Intersects", ext="Contains")
                        globalg.add_edge(n, i, QSR="Intersects", ext="IsWithin")
                    else: #interesect, i.e.., overlap or touch, without containment
                        globalg.add_edge(i, n, QSR="Intersects", ext="TouchesOrOverlaps")
                        globalg.add_edge(n, i, QSR="Intersects", ext="TouchesOrOverlaps")

                if ops[4] is True: #bottom hs of g1 intersects with g2

                    if intersects is True: # with intersection, g1 is on
                        globalg.add_edge(i, n, QSR="IsOn")
                    else: # without intersection, it g1 is above
                        globalg.add_edge(i, n, QSR="IsAbove")
                    #in both cases g2 is below
                    globalg.add_edge(n, i,QSR="IsBelow")

                if ops[5] is True: # top hs of g1 intersects with g2
                    if intersects is True: # with intersection, g2 is on
                        globalg.add_edge(n, i,QSR="IsOn")
                    else: # without intersection, it g2 is above
                        globalg.add_edge(n, i,QSR="IsAbove")
                    #in both cases g1 is below
                    globalg.add_edge(i, n, QSR="IsBelow")

                if any(ops[6:]) is True: #g1 next to g2 and viceversa, with front, back, l/r specialisations

                    if ops[6] is True: #left hs of g1 and g2
                        globalg.add_edge(i, n, QSR="IsNextTo", ext="IsRightOf")
                        globalg.add_edge(n, i, QSR="IsNextTo", ext="IsLeftOf")
                    if ops[7] is True:#right hs of g1 and g2
                        globalg.add_edge(i, n, QSR="IsNextTo", ext="IsLeftOf")
                        globalg.add_edge(n, i, QSR="IsNextTo", ext = "IsRightOf")

                    if ops[8] is True: #front hs of g1 with g2
                        globalg.add_edge(i, n, QSR="IsNextTo", ext="IsBehind")
                        globalg.add_edge(n, i, QSR="IsNextTo", ext="IsInFrontOf")

                    if ops[9] is True: # back hs of g1 with g2
                        globalg.add_edge(i, n, QSR="IsNextTo", ext="IsInFrontOf")
                        globalg.add_edge(n, i, QSR="IsNextTo", ext="IsBehind")

            if not globalg.has_edge(i, n):# by default, the object is a neighbour if in ops
                #fallback to near rel
                globalg.add_edge(i, n, QSR="Near")
                globalg.add_edge(n, i, QSR="Near")

        return globalg


