"""
Interface methods for spatial database
"""
import os
import itertools
import psycopg2
from psycopg2 import Error
import keyring # used for more secure pw storage
from collections import OrderedDict


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


def create_boxes(dbobj, sf=2.0):
    """Creation of min oriented bounding box, contextualised bounding box
    and six halfspaces, for each object/spatialRegion
    """
    # for all objects in spatial DB
    # Find min oriented 3D box bounding of polyhedral surface
    dbobj.cursor.execute('SELECT object_id, ST_OrientedEnvelope(projection_2d), ST_ZMin(object_polyhedral_surface),'
                   ' ST_ZMax(object_polyhedral_surface) FROM single_snap;')
    query_res = [(str(r[0]), str(r[1]), float(r[2]), float(r[3])) for r in dbobj.cursor.fetchall()]
    for id_, envelope, zmin, zmax in query_res:
        height = zmax - zmin
        # ST_Translate is needed here because the oriented envelope is projected on XY so we also need to translate
        # everything up by the original height after extruding in this case
        up1_mask = 'UPDATE single_snap SET bbox = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s)' \
                   ' WHERE object_id = %s;'
        dbobj.cursor.execute(up1_mask, (envelope, str(height), zmin, id_))
        dbobj.connection.commit()

        # TODO replace 0,0 below with robot XY read from a table
        # Derive CBB
        dbobj.cursor.execute(
            'SELECT ST_Angle(ST_MakeLine(ST_MakePoint(0, 0), ST_Centroid(ST_OrientedEnvelope(projection_2d))), '
            'ST_MakeLine(ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(projection_2d)), 1), '
            'ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(projection_2d)), 2))) '
            'FROM single_snap '
            'WHERE object_id = \'' + id_ + '\';')

        dbobj.connection.commit()

        angle = dbobj.cursor.fetchone()[0]

        up2_mask = 'UPDATE single_snap SET cbb = ST_Rotate(bbox, %s,' \
                   ' ST_Centroid(ST_OrientedEnvelope(projection_2d)))' \
                   ' WHERE object_id = %s;'
        dbobj.cursor.execute(up2_mask, (str(angle), id_))
        dbobj.connection.commit()

        # Derive the six halfspaces, based on scaling factor sf

        # top and bottom ones based on MinOriented, i.e., extruded again from oriented envelope
        up_topbtm = 'UPDATE single_snap SET  tophsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s),'\
                    ' bottomhsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s)'\
                   ' WHERE object_id = %s;'
        dbobj.cursor.execute(up_topbtm, (envelope, str(height*sf), zmax, envelope, str(height * sf), zmin-height*sf, id_))

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

        dbobj.cursor.execute(q_hs, (str(sf),str(sf), id_))
        q_res = dbobj.cursor.fetchall() # for each object, 2 rows by 3 colums (i.e., 4 halfspaces + base of cbb repeated twice)

        # Interpret what is L/R/front/back among those boxes
        # TODO replace 0,0 below with robot XY read from a table
        dmin = float('inf')
        amax = float('-inf')
        allfbs = [q[1] for q in q_res]
        alllrs = [q[0] for q in q_res]
        for lr, fb, base in q_res:
            # front is the nearest one to robot position
            qdis = 'SELECT St_Distance(St_MakePoint(0,0), St_Centroid(St_GeomFromEWKT(%s)))'
            dbobj.cursor.execute(qdis, (fb,))
            qdisr = dbobj.cursor.fetchone()[0]
            if qdisr < dmin:
                fronths = fb
            # Left one has the biggest angle with robot position and base centroid (St_Angle is computed clockwise)
            qang = 'SELECT St_Angle(St_MakeLine(St_MakePoint(0,0),St_Centroid(%s)), St_MakeLine(St_MakePoint(0,0)'\
                        ',St_Centroid(St_GeomFromEWKT(%s))))'
            dbobj.cursor.execute(qang, (base,lr))
            qangr = dbobj.cursor.fetchone()[0]
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
        dbobj.cursor.execute(up_others,
                                (lefths, str(height * sf), zmin, righths, str(height * sf), zmin,
                                 fronths,str(height * sf), zmin, backhs,str(height * sf), zmin, id_))
        dbobj.connection.commit()
    """Just for debugging/ visualize the 3D geometries we have just constructed"""
    # return XML representation for 3D web visualizer
    """dbobj.cursor.execute('SELECT ST_AsX3D(bbox), ST_AsX3D(cbb), ST_AsX3D(tophsproj)  FROM single_snap;')
    with open(os.path.join(os.environ['HOME'], 'hsdump.txt'), 'w') as outd:
        for r in dbobj.cursor.fetchall():
            outd.write(r[0] + ',' + r[1] + r[2] + '\n')
    """

def retrieve_ids_ord(session,timestamp):
    timestamp = '2020-05-15-11-02-54_646' #TODO remove when running on full set
    tmp_conn, tmp_cur = session
    tmp_cur.execute('SELECT object_id, ST_Volume(bbox) as v FROM single_snap '\
                    'WHERE object_id LIKE %s '\
                    'ORDER BY v DESC', (timestamp+'%',)) #use string pattern matching to check only first part of ID
    # & order objects by volume, descending, to identify reference objects
    res = tmp_cur.fetchall()
    return OrderedDict(res) #preserve ordering in dict #{k[0]: {} for k in res}  # init with obj ids

def find_neighbours(session, ref_id, ordered_objs,T=2):
    """T = distance threshold to find neighbours, defaults to 2 units in the SRID of spatial DB"""
    #Find nearby objects which are also smaller in the ordering
    i = list(ordered_objs.keys()).index(ref_id)
    candidates = list(ordered_objs.keys())[i+1:] #candidate figure objects, i.e., smaller
    # Which ones are also nearby?
    tmp_conn, tmp_cur = session
    tmp_cur.execute('SELECT object_id FROM single_snap'\
                    ' WHERE ST_3DDWithin(bbox, '\
                    '(SELECT bbox FROM single_snap '\
                    'WHERE object_id = %s), %s) '\
                    'AND object_id != %s', (ref_id,str(T),ref_id))

    nearby = [t[0] for t in tmp_cur.fetchall()]
    figures = [id_ for id_ in nearby if id_ in candidates]
    return figures

def extract_QSR(session, ref_id, figure_objs, qsr_graph, D=1.):
    """
    Expects qsrs to be collected as nx.MultiDGraph
    D is the space granularity as defined in the paper"""
    tmp_conn, tmp_cur = session
    for figure_id in figure_objs:
        if not qsr_graph.has_node(figure_id): #add new node if not already there
            qsr_graph.add_node(figure_id)
        #Use postGIS for deriving truth values of base operators
        tmp_cur.execute('SELECT ST_3DDWithin(fig.bbox,reff.bbox, 0),'\
		                ' ST_3DIntersects(fig.bbox,reff.bbox),'\
                        ' ST_Volume(St_3DIntersection(fig.bbox,reff.bbox)),' \
		                ' ST_Volume(fig.bbox),'\
		                ' ST_3DIntersects(fig.bbox,reff.tophsproj),ST_3DIntersects(fig.bbox,reff.bottomhsproj),'\
		                ' ST_3DIntersects(fig.bbox,reff.lefthsproj),ST_3DIntersects(fig.bbox,reff.righthsproj),'\
		                ' ST_3DIntersects(fig.bbox,reff.fronthsproj), ST_3DIntersects(fig.bbox,reff.backhsproj),'\
		                ' St_Volume(ST_3DIntersection(ST_Scale(ST_3DIntersection(fig.bbox,reff.bbox),1.00001,1.00001,1.00001), fig.bbox)),'\
		                ' St_Volume(ST_3DIntersection(ST_Scale(ST_3DIntersection(fig.bbox,reff.bbox),1.00001,1.00001,1.00001), reff.bbox))'\
                        ' from single_snap as reff, single_snap as fig'\
                        ' WHERE reff.object_id = %s'\
                        ' AND fig.object_id = %s', (ref_id,figure_id))
        #Unpack results and infer QSR predicates
        res = tmp_cur.fetchone()

        # Relations are all directed from figure to reference
        if res[0] is True: qsr_graph.add_edge(figure_id,ref_id, QSR='touches')
        if res[1] is True: qsr_graph.add_edge(figure_id,ref_id, QSR='intersects')

        if res[1] is True and (res[3]-res[2]) <= D: #if volume of intersection very close to volume of smaller object, smaller object is completely contained
            qsr_graph.add_edge(figure_id, ref_id, QSR='completely_contained')
        elif res[1] is True and (res[3]-res[2])> D and res[10]<res[11]: #intersect but only partially In
            qsr_graph.add_edge(figure_id, ref_id, QSR='partially_contained')

        if res[4] is True: qsr_graph.add_edge(figure_id, ref_id, QSR='above')
        if res[5] is True: qsr_graph.add_edge(figure_id, ref_id, QSR='below')
        if res[6] is True: qsr_graph.add_edge(figure_id, ref_id, QSR='leftOf')
        if res[7] is True: qsr_graph.add_edge(figure_id, ref_id, QSR='rightOf')
        if res[6] is True or res[7] is True: qsr_graph.add_edge(figure_id, ref_id, QSR='beside')
        if res[8] is True: qsr_graph.add_edge(figure_id, ref_id, QSR='inFrontOf')
        if res[9] is True: qsr_graph.add_edge(figure_id, ref_id, QSR='behind')

        #Infer more complex QSR by combining the base ones (e.g., touch and above --> on top of)
        if res[0] is True and res[4] is True: qsr_graph.add_edge(figure_id, ref_id, QSR='onTopOf')
        # to test the other cases of ON (affixedOn and leanOn) we need all base QSRs with other objects gathered first

        if not qsr_graph.has_edge(figure_id, ref_id): # If not special rel, by default/definition, they are neighbours
            qsr_graph.add_edge(figure_id, ref_id, QSR='near')

    return qsr_graph


def infer_special_ON(local_graph):
    """Iterates only over QSRs in current image
    but propagates new QSRs found to global graph"""
    for node1 in local_graph.nodes():
        # if obj1 touches or is touched by obj2
        t = [(f,ref,r) for f,ref,r in local_graph.out_edges(node1, data=True) if r['QSR']=='touches']
        is_t = [(f,ref,r) for f,ref,r in local_graph.in_edges(node1, data=True) if r['QSR']=='touches']
        is_a = [f for f,_,r in local_graph.in_edges(node1, data=True) if r['QSR']=='below'] #edges where obj1 is reference and figure objects are below it

        if len(t)==0 and len(is_t)==0: continue #skip
        elif len(t)==1 and len(is_t)==0: #exactly one touch relation
            #where obj1 is fig, obj2 is ref
            node2 = t[0][1]
            l = [k for k in local_graph.get_edge_data(node1,node2) if local_graph.get_edge_data(node1,node2,k)['QSR']=='above']
            if len(l)==0: # obj1 is not above obj2
                local_graph.add_edge(node1, node2, QSR='affixedOn') #then obj1 is affixed on obj2
        elif len(t) ==0 and len(is_t)==1: continue # skip, will be added later when obj is found as fig in for loop
        else: # touches/is touched by more than one object
            #consider those where obj1 is figure
            nodes2 = [tr[1] for tr in t]
            for node2 in nodes2:
                others_below = [n for n in is_a if n!=node2 and n in nodes2]
                l = [k for k in local_graph.get_edge_data(node1, node2) if
                     local_graph.get_edge_data(node1, node2, k)['QSR'] == 'above']
                lb = [k for k in local_graph.get_edge_data(node1, node2) if
                     local_graph.get_edge_data(node1, node2, k)['QSR'] == 'below']
                if len(l)==0 and lb==0 \
                        and len(others_below)>0: #and there is at least an o3 different from o2 which is below o1
                    local_graph.add_edge(node1, node2, QSR='leansOn') # then o1 leans on o2
    return local_graph