"""
Interface methods for spatial database
"""
import itertools
import psycopg2
from psycopg2 import Error
import keyring # used for more secure pw storage
import networkx as nx
import matplotlib.pyplot as plt

def connect_DB(user,dbname):
    try:
        conn = psycopg2.connect(dbname=dbname, user=user, password=keyring.get_password(dbname,user))
        cur = conn.cursor()
        print(conn.get_dsn_parameters(), "\n")
        cur.execute("SELECT version();")
        record = cur.fetchone()
        print("You are connected to - ", record, "\n")

    except (Exception,Error) as error:
        print("Error while connecting to PostgreSQL", error)
        conn = None
    return conn,cur


def disconnect_DB(connection,cursor):
    cursor.close()
    connection.close()
    print("Closed PostgreSQL connection")


def create_VG_table(cursor):
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS VG_RELATIONS("
        "relation_id serial PRIMARY KEY,"
        "predicate_name varchar NOT NULL,"
        "predicate_aliases varchar,"
        "subject_polygon geometry,"
        "object_top_projection geometry);")

        #"ST_Overlaps boolean,"
        #"ST_Touches boolean,"
        #"ST_Within boolean,"
        #"ST_Contains boolean);")
    print("DB table VG_RELATIONS created or exists already")


def add_VG_row(cursor,qargs):
    rel_id, pred, aliases, sub_coords, obj_coords = qargs
    spoly_str = coords2polygon(sub_coords)
    opoly_str = coords2polygon(obj_coords)

    cursor.execute("""INSERT INTO VG_RELATIONS(relation_id, predicate_name, predicate_aliases,subject_polygon,object_top_projection)
                   VALUES (%s,%s,%s,ST_GeomFromText(%s),ST_GeomFromText(%s));
                   """,(rel_id,pred,str(aliases),spoly_str,opoly_str))

    print("Relation datapoint %s added to VG_RELATIONS" % rel_id)

def compute_spatial_op(cursor, current_row_id):
    #Computes spatial operations and concurrently updates table column with result

    cursor.execute("""
            SELECT ST_Overlaps(subject_polygon, object_top_projection)
                                FROM VG_RELATIONS
                                WHERE relation_id=%s;""", (current_row_id,))

    overlaps = cursor.fetchone()[0]

    cursor.execute("""
                SELECT ST_Touches(subject_polygon, object_top_projection)
                                    FROM VG_RELATIONS
                                    WHERE relation_id=%s;""", (current_row_id,))
    touches = cursor.fetchone()[0]

    """
    cursor.execute(\"""
                    UPDATE VG_RELATIONS
                    SET ST_Within =
                    (SELECT ST_Within(subject_polygon, object_top_projection)
                                        FROM VG_RELATIONS
                                        WHERE relation_id=%s)
                    WHERE relation_id=%s;\""", (current_row_id, current_row_id))
    within = cursor.fetchone()[0]
    cursor.execute(\"""
                    UPDATE VG_RELATIONS
                    SET ST_Contains =
                    (SELECT ST_Contains(subject_polygon, object_top_projection)
                                        FROM VG_RELATIONS
                                        WHERE relation_id=%s)
                    WHERE relation_id=%s;\""", (current_row_id, current_row_id))
    contains = cursor.fetchone()[0]"""
    return overlaps, touches

def coords2polygon(coord_list):
    """Formats list of 2D coordinate points
    as string defining Polygon in PostgreSQL"""
    coords_ = [str(x) + " " + str(y) for x, y in coord_list]
    return "POLYGON(("+",".join(t for t in coords_)+"))"

def coords2polyhedral(vertex_list):
    """# Given list of 8 vertices as
    # [ (Xmin, Ymin, Zmin)
        (Xmin, Ymax, Zmin)
        (Xmin, Ymax, Zmax)
        (Xmin, Ymin, Zmax)
        (Xmax, Ymin, Zmin)
        (Xmax, Ymax, Zmin)
        (Xmax, Ymax, Zmax)
        (Xmax, Ymin, Zmax) ]
    formats it as polyhedral surface string
    """
    # first 2 faces are given by input order
    coords_ = vertex_list[:4]+ list(vertex_list[0]) #close ring
    #coords_.extend(vertex_list[4:]+list(vertex_list[4]))
    #coords_.extend(vertex_list[:2]+vertex_list[:5]+list(vertex_list[0]))
    coords_.extend(vertex_list[1:3] + vertex_list[5:7] + list(vertex_list[1]))
    coords_.extend(vertex_list[3:5] + vertex_list[6:8] + list(vertex_list[3]))
    coords_.extend(list(vertex_list[0]) + list(vertex_list[3]) +list(vertex_list[4]) + list(vertex_list[7]) + list(vertex_list[0]))
    #TODO complete this
    return "POLYHEDRAL SURFACE Z(("+",".join(t for t in coords_)+"))"

def minmax_to_vertex(minmax):
    # Expects list of 3D extent of order [ST_XMin, ST_YMin, ST_ZMin, ST_XMax,ST_YMax, ST_ZMax]
    # returns minimal list of 8 vertices defining box 3D
    return [
        [minmax[0], minmax[1], minmax[2]],
        [minmax[0], minmax[4], minmax[2]],
        [minmax[0], minmax[4], minmax[5]],
        [minmax[0], minmax[1], minmax[5]],
        [minmax[3], minmax[1], minmax[2]],
        [minmax[3], minmax[4], minmax[2]],
        [minmax[3], minmax[4], minmax[5]],
        [minmax[3], minmax[1], minmax[5]]
        ]

def query_all_bboxes(reasoner):
    # for all objects in spatial DB
    reasoner.cursor.execute("""SELECT object_id FROM test_map""")
    all_ids = [str(i[0]) for i in reasoner.cursor.fetchall()]

    box_coords ={}
    for i in all_ids:
        # Find min 3D box bounding polyhedral surface (ST_3DExtent)
        # and  extract bbox vertex coords
        reasoner.cursor.execute("""
            SELECT ST_XMin(g1), ST_YMin(g1), ST_ZMin(g1),
            ST_XMax(g1), ST_YMax(g1), ST_ZMax(g1)
            FROM (
                    SELECT ST_3DExtent(object_polyhedral_surface) as g1
                    FROM test_map
                    WHERE object_id = %s
        ) as xyz;""", (i,))

        minmax = reasoner.cursor.fetchall()[0]
        box_vertices= minmax_to_vertex(minmax)
        box_coords[i] ={}
        box_coords[i]["vertices"] = box_vertices
        box_coords[i]["x_extent"] = minmax[3] - minmax[0]
        box_coords[i]["y_extent"] = minmax[4] - minmax[1]
        box_coords[i]["z_extent"] = minmax[5] - minmax[2]

    return box_coords

def plot_graph(G):

    nx.draw(G)
    plt.draw()
    plt.show()

def extract_QSRs(reasoner, bbox_dict, search_radius=6,topological_rels=["St_Overlaps"\
                            ,"ST_Touches","ST_Within"]):
    semmap = nx.MultiDiGraph()
    semmap.add_nodes_from(bbox_dict.keys())  # add one node per object_id

    for i in bbox_dict.keys():
        # Transform halfspace projections from minmax list to vertex coords
        # and convert vertex list to polyhedral surface WKT
        top_hs = coords2polyhedral(minmax_to_vertex(bbox_dict[i]["top_hs"]))
        btm_hs = coords2polyhedral(minmax_to_vertex(bbox_dict[i]["btm_hs"]))
        front_hs = coords2polyhedral(minmax_to_vertex(bbox_dict[i]["front_hs"]))
        back_hs = coords2polyhedral(minmax_to_vertex(bbox_dict[i]["back_hs"]))
        left_hs = coords2polyhedral(minmax_to_vertex(bbox_dict[i]["left_hs"]))
        right_hs = coords2polyhedral(minmax_to_vertex(bbox_dict[i]["right_hs"]))

        #Alter table to add halfspace projection data
        reasoner.connection.commit() # make new data visible for next queries

        # Find all nearby points within given radius
        # TODO find intersections with halfspace projections to add above, below, L/R, front, behind
        reasoner.cursor.execute("""
                        SELECT g2.object_id, ST_Overlaps(g1.object_polyhedral_surface, g2.object_polyhedral_surface),
                        ST_Touches(g1.object_polyhedral_surface, g2.object_polyhedral_surface), 
                        ST_Within(g1.object_polyhedral_surface, g2.object_polyhedral_surface)
                        FROM test_map AS g1, test_map AS g2
                        WHERE ST_3DDWithin(g1.object_polyhedral_surface, g2.object_polyhedral_surface, %s)
                        AND g1.object_id = %s AND g2.object_id != %s
                        ;""", (str(search_radius), i, i))

        results = [(r[0], list(r[1:])) for r in reasoner.cursor.fetchall()]  # [i for i in reasoner.cursor.fetchall()]

        for n, ops in results:
            if not semmap.has_edge(i, n):  # if object and neighbour not connected already
                # avoiding duplicates and reducing computational cost
                tgt_rels = [name for k, name in enumerate(topological_rels) if ops[k]]
                # add edges between node pairs based on results
                # True = edge, False = no edge
                for index, name in enumerate(tgt_rels):
                    semmap.add_edge(i, n)
                    semmap.add_edge(n, i)  # topological are bidirectional
                    semmap[i][n][index]["name"] = name
                    if name == "ST_Within":
                        semmap[n][i][index]["name"] = "ST_Contains"  # contains is inverse of within
                    else:
                        semmap[n][i][index]["name"] = name
    return semmap