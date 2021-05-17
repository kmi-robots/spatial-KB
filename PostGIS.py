"""
Interface methods for spatial database
"""
import os
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
        conn, cur = None, None
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

    return overlaps, touches

def coords2polygon(coord_list):
    """Formats list of 2D coordinate points
    as string defining Polygon in PostgreSQL"""
    coords_ = [str(x) + " " + str(y) for x, y in coord_list]
    return "POLYGON(("+",".join(t for t in coords_)+"))"


def create_boxes(reasoner, sf=2.0):
    """Creation of min oriented bounding box, contextualised bounding box
    and six halfspaces, for each object/spatialRegion
    """
    # for all objects in spatial DB
    # Find min oriented 3D box bounding of polyhedral surface
    reasoner.cursor.execute('SELECT object_id, ST_OrientedEnvelope(projection_2d), ST_ZMin(object_polyhedral_surface),'
                   ' ST_ZMax(object_polyhedral_surface) FROM single_snap;')
    query_res = [(str(r[0]), str(r[1]), float(r[2]), float(r[3])) for r in reasoner.cursor.fetchall()]
    for id_, envelope, zmin, zmax in query_res:
        height = zmax - zmin
        # ST_Translate is needed here because the oriented envelope is projected on XY so we also need to translate
        # everything up by the original height after extruding in this case
        up1_mask = 'UPDATE single_snap SET bbox = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s)' \
                   ' WHERE object_id = %s;'
        reasoner.cursor.execute(up1_mask, (envelope, str(height), zmin, id_))
        reasoner.connection.commit()

        # TODO replace 0,0 below with robot XY read from a table
        # Derive CBB
        reasoner.cursor.execute(
            'SELECT ST_Angle(ST_MakeLine(ST_MakePoint(0, 0), ST_Centroid(ST_OrientedEnvelope(projection_2d))), '
            'ST_MakeLine(ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(projection_2d)), 1), '
            'ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(projection_2d)), 2))) '
            'FROM single_snap '
            'WHERE object_id = \'' + id_ + '\';')

        reasoner.connection.commit()

        angle = reasoner.cursor.fetchone()[0]

        up2_mask = 'UPDATE single_snap SET cbb = ST_Rotate(bbox, %s,' \
                   ' ST_Centroid(ST_OrientedEnvelope(projection_2d)))' \
                   ' WHERE object_id = %s;'
        reasoner.cursor.execute(up2_mask, (str(angle), id_))
        reasoner.connection.commit()

        # Derive the six halfspaces, based on scaling factor sf

        # top and bottom ones based on MinOriented, i.e., extruded again from oriented envelope
        up_topbtm = 'UPDATE single_snap SET  tophsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s),'\
                    ' bottomhsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s)'\
                   ' WHERE object_id = %s;'
        reasoner.cursor.execute(up_topbtm, (envelope, str(height*sf), zmax, envelope, str(height * sf), zmin-height*sf, id_))

        # Identify the base of the CBB, i.e., oriented envelope of points with z=zmin
        # Define 4 rectangles from the base and expand 2D
        # TODO replace 0,0 below with robot XY read from a table
        q_hs = 'SELECT St_Rotate(hsX, alpha, St_centroid(aligned_base)), St_Rotate(hsY, alpha, St_centroid(aligned_base)), base '\
                'FROM('\
                    'SELECT (St_Dump(r1)).geom as hsX, (St_Dump(r2)).geom as hsY, alpha, aligned_base, base, w, d '\
                    'FROM('\
                    'SELECT alpha, aligned_base, base, w, d, St_Difference(St_Expand(aligned_base, %s * w, 0),'\
                        'St_Scale(aligned_base, St_MakePoint(1.00001,1.00001), St_Centroid(aligned_base))) as r1,'\
                        'St_Difference(St_Expand(aligned_base, 0, %s * d), St_Scale(aligned_base,'\
                        'St_MakePoint(1.00001,1.00001), St_Centroid(aligned_base))) as r2 '\
                        'FROM( SELECT St_Rotate(base,-alpha, ST_Centroid(base)) as aligned_base, w, d, alpha, base '\
                        'FROM( SELECT base, St_XMax(base) - St_XMin(base) as w, St_YMax(base) - St_YMin(base) as d,'\
                        'St_Angle(St_MakeLine(ST_PointN(ST_ExteriorRing(base),1), ST_PointN(ST_ExteriorRing(base),2)),'\
		                'St_MakeLine(ST_MakePoint(0,0), ST_MakePoint(0,1))) as alpha '\
		                'FROM (SELECT ST_OrientedEnvelope(St_Collect((dbox).geom)) as base '\
	  					'FROM(SELECT St_ZMax(cbb)- St_ZMin(cbb) as h, St_DumpPoints(cbb) as dbox,'\
                                'St_ZMin(cbb) as zmin FROM single_snap '\
		    					'WHERE object_id=%s) as dt '\
	                            'WHERE St_Z((dbox).geom) = zmin)   as basal'\
                        ') as angles'\
                        ') as aligned'\
                        ') as hs'\
                ')as fcheck'

        reasoner.cursor.execute(q_hs, (str(sf),str(sf), id_))
        q_res = reasoner.cursor.fetchall() # for each object, 2 rows by 3 colums (i.e., 4 halfspaces + base of cbb repeated twice)

        # Interpret what is L/R/front/back among those boxes
        # TODO replace 0,0 below with robot XY read from a table
        dmin = float('inf')
        amax = float('-inf')
        allfbs = [q[1] for q in q_res]
        alllrs = [q[0] for q in q_res]
        for lr, fb, base in q_res:
            # front is the nearest one to robot position
            qdis = 'SELECT St_Distance(St_MakePoint(0,0), St_Centroid(St_GeomFromEWKT(%s)))'
            reasoner.cursor.execute(qdis, (fb,))
            qdisr = reasoner.cursor.fetchone()[0]
            if qdisr < dmin:
                fronths = fb
            # Left one has the biggest angle with robot position and base centroid (St_Angle is computed clockwise)
            qang = 'SELECT St_Angle(St_MakeLine(St_MakePoint(0,0),St_Centroid(%s)), St_MakeLine(St_MakePoint(0,0)'\
                        ',St_Centroid(St_GeomFromEWKT(%s))))'
            reasoner.cursor.execute(qang, (base,lr))
            qangr = reasoner.cursor.fetchone()[0]
            if qangr > amax:
                lefths = lr
        backhs = [fb for fb in allfbs if fb!= fronths][0] # the one record which is not the front one will be the back one
        righths = [lr for lr in alllrs if lr!= lefths][0] # similarly for L/R
        # Extrude + Translate 3D & update table with halfspace columns
        up_others = 'UPDATE single_snap SET  lefthsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s),' \
                    ' righthsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s),' \
                    ' fronthsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s),' \
                    ' backhsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s)' \
                    ' WHERE object_id = %s;'
        reasoner.cursor.execute(up_others,
                                (lefths, str(height * sf), zmin, righths, str(height * sf), zmin,
                                 fronths,str(height * sf), zmin, backhs,str(height * sf), zmin, id_))
        reasoner.connection.commit()
    """Just for debugging/ visualize the 3D geometries we have just constructed"""
    # return XML representation for 3D web visualizer
    """reasoner.cursor.execute('SELECT ST_AsX3D(bbox), ST_AsX3D(cbb), ST_AsX3D(tophsproj)  FROM single_snap;')
    with open(os.path.join(os.environ['HOME'], 'hsdump.txt'), 'w') as outd:
        for r in reasoner.cursor.fetchall():
            outd.write(r[0] + ',' + r[1] + r[2] + '\n')
    """

def query_map_neighbours(reasoner, bbox_dict, i, search_radius=6):

        # Find all nearby points within given radius and compute QSRs
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
                        FROM semantic_map AS g1, semantic_map AS g2
                        WHERE ST_3DDWithin(g\s1.object_polyhedral_surface, g2.object_polyhedral_surface, %s)
                        AND g1.object_id = %s AND g2.object_id != %s
                        ;""", (str(search_radius), i, i))

        return [(str(r[0]), list(r[1:])) for r in reasoner.cursor.fetchall()]  # [i for i in reasoner.cursor.fetchall()]

