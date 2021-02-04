"""
Interface methods for spatial database
"""
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
    spoly_str = coords2poly(sub_coords)
    opoly_str = coords2poly(obj_coords)

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

def coords2poly(coord_list):
    """Formats list of 2D coordinate points
    as string defining Polygon in PostgreSQL"""
    coords_ = [str(x) + " " + str(y) for x, y in coord_list]
    return "POLYGON(("+",".join(t for t in coords_)+"))"


def threed_spatial_query(reasoner, search_radius=6, topological_rels=["St_Overlaps"\
                            ,"ST_Touches","ST_Within"]):
    # for all objects in spatial DB
    reasoner.cursor.execute("""SELECT object_id FROM test_map""")
    all_ids = [str(i[0]) for i in reasoner.cursor.fetchall()]

    semmap = nx.MultiDiGraph()
    semmap.add_nodes_from(all_ids) # add one node per object

    for i in all_ids:
        #Find ST relations with all nearby points within given radius
        reasoner.cursor.execute("""
                SELECT g2.object_id, ST_Overlaps(g1.object_polyhedral_surface, g2.object_polyhedral_surface),
                ST_Touches(g1.object_polyhedral_surface, g2.object_polyhedral_surface), 
                ST_Within(g1.object_polyhedral_surface, g2.object_polyhedral_surface)
                FROM test_map AS g1, test_map AS g2
                WHERE ST_3DDWithin(g1.object_polyhedral_surface, g2.object_polyhedral_surface, %s)
                AND g1.object_id = %s AND g2.object_id != %s
                ;""", (str(search_radius),i,i))

        results = [(r[0], list(r[1:])) for r in reasoner.cursor.fetchall()] #[i for i in reasoner.cursor.fetchall()]

        for n,ops in results:
            if not semmap.has_edge(i,n):    # if object and neighbour not connected already
                                            # avoiding duplicates and reducing computational cost
                tgt_rels = [name for k,name in enumerate(topological_rels) if ops[k]]
                # add edges between node pairs based on results
                # True = edge, False = no edge
                for index, name in enumerate(tgt_rels):
                    semmap.add_edge(i, n)
                    semmap.add_edge(n, i)  # topological are bidirectional
                    semmap[i][n][index]["name"] = name
                    if name =="ST_Within":
                        semmap[n][i][index]["name"] = "ST_Contains" #contains is inverse of within
                    else:
                        semmap[n][i][index]["name"] = name

        print(semmap.edges(data=True))
        #plot_graph(semmap)
        reasoner.cursor.execute("""
        SELECT ST_x(geom), ST_y(geom), ST_z(geom) FROM (
        SELECT (ST_DumpPoints(object_polyhedral_surface)).geom FROM test_map
        WHERE object_id=%s
        ) as xyz;""", (i,))
        box_coords = reasoner.cursor.fetchall()
        continue
    return semmap,[]

def plot_graph(G):

    nx.draw(G)
    plt.draw()
    plt.show()