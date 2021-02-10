"""
Interface methods for spatial database
"""
import itertools
import psycopg2
from psycopg2 import Error
import keyring # used for more secure pw storage


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

def minmax_to_polyhedral(minmax):
    """
    Expects list of 3D extent of order [ST_XMin, ST_YMin, ST_ZMin, ST_XMax,ST_YMax, ST_ZMax]
    formats it as polyhedral surface string
    if order does not matter, vertex combinations can be found dynamically
    """
    all_face_vs = []
    for i,nm in enumerate(['xmin', 'ymin', 'zmin', 'xmax','ymax', 'zmax']):
        #keep one dimension fixed and permutate all the others
        r_minmax = minmax.copy()
        r_minmax.remove(minmax[i])
        if i<3: r_minmax.remove(minmax[i+3])# e.g., remove all xs
        else: r_minmax.remove(minmax[i-3]) #avoid index out of range
        A = [r_minmax[0], r_minmax[2]]
        B = [r_minmax[1], r_minmax[3]]
        #and convert to formatted string
        if nm[0] =='x':
            vs = [" ".join((str(minmax[i]),str(p[0]),str(p[1]))) for p in itertools.product(A,B)]
        elif nm[0] == 'y':
            vs = [" ".join((str(p[0]),str(minmax[i]),str(p[1]))) for p in itertools.product(A, B)]
        else: #z
            vs = [" ".join((str(p[0]),str(p[1]),str(minmax[i]))) for p in itertools.product(A, B)]
        vs.append(vs[0]) #lastly, close ring, i.e., repeat one coord triple
        all_face_vs.append(vs)

    s = ["(("+",".join(v for v in face) +"))" for face in all_face_vs]
    return "POLYHEDRALSURFACE Z("+",".join(f for f in s)+")"

def ordered_minmax_to_polyhedral(minmax):
    #Order of drawing faces(clockwise, anticlockwise) counts in postgre
    # vertex ordering is hardcoded
    xmin_, ymin_, zmin_, xmax_,ymax_, zmax_ = minmax
    faces= [ [[xmin_,ymin_,zmin_],[xmin_,ymax_,zmin_],[xmax_,ymax_,zmin_],[xmax_,ymin_,zmin_],[xmin_,ymin_,zmin_]], #clockwise
            [[xmin_,ymin_,zmax_],[xmax_,ymin_,zmax_],[xmax_,ymax_,zmax_],[xmin_,ymax_,zmax_],[xmin_,ymin_,zmax_]],  #counterclockwise
            [[xmin_,ymin_,zmin_],[xmin_,ymin_,zmax_],[xmin_,ymax_,zmax_],[xmin_,ymax_,zmin_],[xmin_,ymin_,zmin_]],  #clockwise
            [[xmin_,ymax_,zmin_],[xmin_,ymax_,zmax_],[xmax_,ymax_,zmax_],[xmax_,ymax_,zmin_],[xmin_,ymax_,zmin_]],  #counterclockwise
            [[xmax_,ymax_,zmin_],[xmax_,ymax_,zmax_],[xmax_,ymin_,zmax_],[xmax_,ymin_,zmin_],[xmax_,ymax_,zmin_]],  #counterclockwise
            [[xmax_,ymin_,zmin_],[xmax_,ymin_,zmax_],[xmin_,ymin_,zmax_],[xmin_,ymin_,zmin_],[xmax_,ymin_,zmin_]]   #clockwise
            ]

    s = []
    for face in faces:
        t_face = []
        for coord_triple in face:
            coord_triple =[str(c) for c in coord_triple]
            t_triple = " ".join(coord_triple)
            t_face.append(t_triple)

        t_face = "(("+",".join(t_face)+"))"
        s.append(t_face)
    return "POLYHEDRALSURFACE Z("+",".join(f for f in s)+")"

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
    # Find min 3D box bounding polyhedral surface (ST_3DExtent)
    # and  extract bbox vertex coords
    reasoner.cursor.execute("""SELECT object_id FROM test_map""")
    all_ids = [str(i[0]) for i in reasoner.cursor.fetchall()]

    box_coords ={}
    for i in all_ids:
        reasoner.cursor.execute("""
            SELECT ST_XMin(g1), ST_YMin(g1), ST_ZMin(g1),
            ST_XMax(g1), ST_YMax(g1), ST_ZMax(g1)
            FROM (  SELECT ST_3DExtent(object_polyhedral_surface) as g1
                    FROM test_map
                    WHERE object_id = %s
        ) as xyz;""", (i,))

        minmax = reasoner.cursor.fetchall()[0]
        box_vertices = minmax_to_vertex(minmax)
        core_obj = ordered_minmax_to_polyhedral(minmax) #add ordered boundingbox polygon to table
        reasoner.cursor.execute("""
                    UPDATE test_map 
                    SET bbox = ST_GeomFromText(%s)
                    WHERE object_id=%s;""", (core_obj,i))
        reasoner.connection.commit()
        box_coords[i] ={}
        box_coords[i]["vertices"] = box_vertices
        box_coords[i]["minmax"] = minmax
        box_coords[i]["x_extent"] = minmax[3] - minmax[0]
        box_coords[i]["y_extent"] = minmax[4] - minmax[1]
        box_coords[i]["z_extent"] = minmax[5] - minmax[2]
    return box_coords


def query_map_neighbours(reasoner, bbox_dict, i, search_radius=6):
        # Transform halfspace projections from minmax list to textual format expected by Postgre
        top_hs = ordered_minmax_to_polyhedral(bbox_dict[i]["top_hs"])#minmax_to_polyhedral(bbox_dict[i]["top_hs"])
        btm_hs = ordered_minmax_to_polyhedral(bbox_dict[i]["btm_hs"])
        front_hs = ordered_minmax_to_polyhedral(bbox_dict[i]["front_hs"])
        back_hs = ordered_minmax_to_polyhedral(bbox_dict[i]["back_hs"])
        left_hs = ordered_minmax_to_polyhedral(bbox_dict[i]["left_hs"])
        right_hs = ordered_minmax_to_polyhedral(bbox_dict[i]["right_hs"])

        #Alter table to add halfspace projection data
        reasoner.cursor.execute("""
            UPDATE test_map 
            SET bottomhsproj = ST_GeomFromText(%s),
                tophsproj = ST_GeomFromText(%s),
                lefthsproj = ST_GeomFromText(%s),
                righthsproj = ST_GeomFromText(%s),
                fronthsproj = ST_GeomFromText(%s),
                backhsproj = ST_GeomFromText(%s)
            WHERE object_id=%s
        ;""", (btm_hs, top_hs, left_hs, right_hs,front_hs,back_hs,i))
        reasoner.connection.commit() # make new data visible for next queries

        # Find all nearby points within given radius and compute
        # topological operators for each obj pair
        # - Interesection between the two objects (replaces 2D overlap and 2D touch)
        # - Volume comparisons (e.g., if the volume of the intersection equals the volume of obj1, then obj1 is fully contained in obj2)
        # Directional QSRs
        # - Intersections between obj2 and halfspace projections of obj1
        reasoner.cursor.execute("""
                        SELECT g2.object_id, ST_3DIntersects(g1.bbox, g2.bbox),
                        ST_Volume(ST_MakeSolid(g1.bbox)), ST_Volume(ST_MakeSolid(g2.bbox)), 
                        ST_Volume(ST_MakeSolid(ST_3DIntersection(g1.bbox, g2.bbox))),
                        ST_3DIntersects(g1.bottomhsproj,g2.bbox), 
                        ST_3DIntersects(g1.tophsproj,g2.bbox),
                        ST_3DIntersects(g1.lefthsproj,g2.bbox), 
                        ST_3DIntersects(g1.righthsproj,g2.bbox),
                        ST_3DIntersects(g1.fronthsproj,g2.bbox), 
                        ST_3DIntersects(g1.backhsproj,g2.bbox)
                        FROM test_map AS g1, test_map AS g2
                        WHERE ST_3DDWithin(g1.object_polyhedral_surface, g2.object_polyhedral_surface, %s)
                        AND g1.object_id = %s AND g2.object_id != %s
                        ;""", (str(search_radius), i, i))

        return [(str(r[0]), list(r[1:])) for r in reasoner.cursor.fetchall()]  # [i for i in reasoner.cursor.fetchall()]

