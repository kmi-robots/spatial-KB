"""
Interface methods for spatial database
"""
import os
import itertools
import psycopg2
from psycopg2 import Error
import keyring # used for more secure pw storage
from collections import OrderedDict
# from utils.x3domviz import generate_html_viz

def connect_DB(user,dbname):
    try:
        conn = psycopg2.connect(dbname=dbname, user=user, password=keyring.get_password(dbname,user))
        cur = conn.cursor()
        #print(conn.get_dsn_parameters(), "\n")
        #cur.execute("SELECT version();")
        #record = cur.fetchone()
        #print("You are connected to - ", record, "\n")
    except (Exception,Error) as error:
        print("Error while connecting to PostgreSQL", error)
        conn, cur = None, None
    return conn,cur


def disconnect_DB(connection,cursor):
    cursor.close()
    connection.close()
    # print("Closed PostgreSQL connection")

def create_VG_table(cursor):
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS VG_RELATIONS("
        "relation_id serial PRIMARY KEY,"
        "predicate_name varchar NOT NULL,"
        "subject_polygon geometry,"
        "object_top_projection geometry);")
    print("DB table VG_RELATIONS created or exists already")

def add_VG_row(cursor,qargs):
    rel_id, pred, sub_coords, obj_coords = qargs
    spoly_str = coords2polygon(sub_coords)
    opoly_str = coords2polygon(obj_coords)

    cursor.execute("""INSERT INTO VG_RELATIONS(relation_id, predicate_name, subject_polygon,object_top_projection)
                   VALUES (%s,%s,ST_GeomFromText(%s),ST_GeomFromText(%s));
                   """,(rel_id,pred,spoly_str,opoly_str))

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


def create_boxes(dbobj, sf=1.2):
    """Creation of min oriented bounding box, contextualised bounding box
    and six halfspaces, for each object/spatialRegion
    """
    # for all objects in spatial DB
    # except crops without depth data associated , i.e., if obj polyhedral surface or projection2d is null
    # Find min oriented 3D box bounding of polyhedral surface
    dbobj.cursor.execute('SELECT object_id, ST_OrientedEnvelope(projection_2d), ST_ZMin(object_polyhedral_surface),'
                   ' ST_ZMax(object_polyhedral_surface) FROM semantic_map '
                         'where object_polyhedral_surface is not null;')
    # Also filter out blacklisted objects that were not run with size reasoning
    blacklist = ['246928_6', '655068_5', '655068_6']
    query_res = [(str(r[0]), str(r[1]), float(r[2]), float(r[3])) for r in dbobj.cursor.fetchall() \
                 if '_'.join(str(r[0]).split('_')[1:]) not in blacklist]
    #images to process
    tbprocessed = [q[0] for q in query_res]

    for id_, envelope, zmin, zmax in query_res:
        height = zmax - zmin
        # ST_Translate is needed here because the oriented envelope is projected on XY so we also need to translate
        # everything up by the original height after extruding in this case
        up1_mask = 'UPDATE semantic_map SET bbox = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s)' \
                   ' WHERE object_id = %s;'
        dbobj.cursor.execute(up1_mask, (envelope, str(height), zmin, id_))
        dbobj.connection.commit()

        # Derive CBB
        dbobj.cursor.execute(
            'SELECT ST_Angle(ST_MakeLine(robot_position, ST_Centroid(ST_OrientedEnvelope(projection_2d))), '
            'ST_MakeLine(ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(projection_2d)), 1), '
            'ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(projection_2d)), 2))) '
            'FROM semantic_map '
            'WHERE object_id = \'' + id_ + '\';')

        dbobj.connection.commit()

        angle = dbobj.cursor.fetchone()[0]

        up2_mask = 'UPDATE semantic_map SET cbb = ST_Rotate(bbox, %s,' \
                   ' ST_Centroid(ST_OrientedEnvelope(projection_2d)))' \
                   ' WHERE object_id = %s;'
        dbobj.cursor.execute(up2_mask, (str(angle), id_))
        dbobj.connection.commit()

        # Derive the six halfspaces, based on scaling factor sf

        # top and bottom ones based on MinOriented, i.e., extruded again from oriented envelope
        up_topbtm = 'UPDATE semantic_map SET  tophsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s),'\
                    ' bottomhsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s)'\
                   ' WHERE object_id = %s;'
        dbobj.cursor.execute(up_topbtm, (envelope, str(height*sf), zmax, envelope, str(height * sf), zmin-height*sf, id_))

        # Identify the base of the CBB, i.e., oriented envelope of points with z=zmin
        # Define 4 rectangles from the base and expand 2D
        q_hs = 'SELECT St_Rotate(hsX, alpha, St_centroid(aligned_base)), St_Rotate(hsY, alpha, St_centroid(aligned_base)), base, robot_position '\
                'FROM('\
                    'SELECT (St_Dump(r1)).geom as hsX, (St_Dump(r2)).geom as hsY, alpha, aligned_base, base, w, d, robot_position '\
                    'FROM('\
                    'SELECT robot_position, alpha, aligned_base, base, w, d, St_Difference(St_Expand(aligned_base, %s * w, 0),'\
                        'St_Scale(aligned_base, St_MakePoint(1.00001,1.00001), St_Centroid(aligned_base))) as r1,'\
                        'St_Difference(St_Expand(aligned_base, 0, %s * d), St_Scale(aligned_base,'\
                        'St_MakePoint(1.00001,1.00001), St_Centroid(aligned_base))) as r2 '\
                        'FROM( SELECT St_Rotate(base,-alpha, ST_Centroid(base)) as aligned_base, w, d, alpha, base, robot_position '\
                        'FROM( SELECT robot_position,base, St_XMax(base) - St_XMin(base) as w, St_YMax(base) - St_YMin(base) as d,'\
                        'St_Angle(St_MakeLine(ST_PointN(ST_ExteriorRing(base),1), ST_PointN(ST_ExteriorRing(base),2)),'\
		                'St_MakeLine(robot_position, ST_MakePoint(ST_X(robot_position),ST_Y(robot_position)+ 1))) as alpha '\
		                'FROM (SELECT ST_OrientedEnvelope(St_Collect((dbox).geom)) as base, robot_position '\
	  					'FROM(SELECT St_DumpPoints(cbb) as dbox, '\
                                'St_ZMin(cbb) as zmin, robot_position FROM semantic_map '\
		    					'WHERE object_id=%s) as dt '\
	                            'WHERE St_Z((dbox).geom) = zmin '\
                                'GROUP BY robot_position )  as basal '\
                        ') as angles'\
                        ') as aligned'\
                        ') as hs'\
                ')as fcheck'

        dbobj.cursor.execute(q_hs, (str(sf),str(sf), id_))
        q_res = dbobj.cursor.fetchall() # for each object, 2 rows by 3 colums (i.e., 4 halfspaces + base of cbb repeated twice)

        # Interpret what is L/R/front/back among those boxes
        all_dis =[]
        all_angles = []
        allfbs = [q[1] for q in q_res]
        alllrs = [q[0] for q in q_res]
        for lr, fb, base,rp in q_res:
            # Distance between robot position and hs centroid
            qdis = 'SELECT St_Distance(%s, St_Centroid(St_GeomFromEWKT(%s)))'
            dbobj.cursor.execute(qdis, (rp,fb,))
            qdisr = dbobj.cursor.fetchone()[0]
            all_dis.append(qdisr)
            # angle between robot position and base centroid (St_Angle is computed clockwise)
            qang = 'SELECT St_Angle(St_MakeLine(%s,St_Centroid(%s)), St_MakeLine(%s '\
                        ',St_Centroid(St_GeomFromEWKT(%s))))'
            dbobj.cursor.execute(qang, (rp,base,rp,lr))
            qangr = dbobj.cursor.fetchone()[0]
            all_angles.append(qangr)
        #front is the nearest one to robot position
        fronths = q_res[all_dis.index(min(all_dis))][1]
        #Left one has the biggest angle with robot position and base centroid
        lefths = q_res[all_angles.index(max(all_angles))][0]
        backhs = [fb for fb in allfbs if fb!= fronths][0] # the one record which is not the front one will be the back one
        righths = [lr for lr in alllrs if lr!= lefths][0] # similarly for L/R
        # Extrude + Translate 3D & update table with halfspace columns
        up_others = 'UPDATE semantic_map SET  lefthsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s),' \
                    ' righthsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s),' \
                    ' fronthsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s),' \
                    ' backhsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s)' \
                    ' WHERE object_id = %s;'
        dbobj.cursor.execute(up_others,
                                (lefths, str(height), zmin, righths, str(height), zmin,
                                 fronths,str(height), zmin, backhs,str(height), zmin, id_))
        dbobj.connection.commit()
    """Just for debugging/ visualize the 3D geometries we have just constructed"""
    # return XML representation for 3D web visualizer
    #tstamp = '2020-05-15-11-13-38_805787' #'2020-05-15-11-24-12_379522'  # remove when running on full set
    #generate_html_viz(dbobj,tstamp)
    return tbprocessed

def retrieve_ids_ord(session,timestamp, vthresh= 10.):
    #now we also filter objects with volume above 10 m3 (clear outliers)
    tmp_conn, tmp_cur = session
    tmp_cur.execute('SELECT object_id, ST_Volume(bbox) as v FROM semantic_map '\
                    'WHERE object_id LIKE %s AND ST_Volume(bbox) <= %s'\
                    'ORDER BY v DESC', (timestamp+'%',vthresh)) #use string pattern matching to check only first part of ID
    # & order objects by volume, descending, to identify reference objects
    res = tmp_cur.fetchall()
    return OrderedDict(res) #preserve ordering in dict #{k[0]: {} for k in res}  # init with obj ids

def find_neighbours(session, ref_id, ordered_objs,T=2):
    """T = distance threshold to find neighbours, defaults to 2 units in the SRID of spatial DB"""
    #Find nearby objects which are also smaller in the ordering
    i = list(ordered_objs.keys()).index(ref_id)
    candidates = list(ordered_objs.keys())[i+1:] #candidate figure objects, i.e., smaller
    # Which ones are nearby?
    tmp_conn, tmp_cur = session
    tmp_cur.execute('SELECT object_id FROM semantic_map'\
                    ' WHERE ST_3DDWithin(bbox, '\
                    '(SELECT bbox FROM semantic_map '\
                    'WHERE object_id = %s), %s) '\
                    'AND object_id != %s', (ref_id,str(T),ref_id))

    nearby = [t[0] for t in tmp_cur.fetchall()]
    return [id_ for id_ in nearby if id_ in candidates] # return only the ones which are both nearby and smaller than

def extract_QSR(session, ref_id, figure_objs, qsr_graph, D=1.0):
    """
    Expects qsrs to be collected as nx.MultiDGraph
    D is the space granularity as defined in the paper"""
    tmp_conn, tmp_cur = session
    for figure_id in figure_objs:
        isAbove = False
        touches = False
        if not qsr_graph.has_node(figure_id): #add new node if not already there
            qsr_graph.add_node(figure_id)
        #Use postGIS for deriving truth values of base operators
        # tmp_conn, tmp_cur = connect_DB(us,dbname)
        try:
            tmp_cur.execute('SELECT ST_3DDWithin(fig.bbox,reff.bbox, 0)'
                            ' from semantic_map as reff, semantic_map as fig'
                            ' WHERE reff.object_id = %s'\
                        ' AND fig.object_id = %s', (ref_id,figure_id))
            res = tmp_cur.fetchone()
            if res[0] is True:
                qsr_graph.add_edge(figure_id, ref_id, QSR='touches')
                touches = True
        except Exception as e:
            print(str(e))
            print("Query too large, server problem raised")
            print(ref_id + " " + figure_id)
            return qsr_graph
        try:
            tmp_cur.execute('SELECT ST_3DIntersects(fig.bbox,reff.tophsproj),ST_3DIntersects(fig.bbox,reff.bottomhsproj)'
                            ' from semantic_map as reff, semantic_map as fig'
                            ' WHERE reff.object_id = %s' \
                            ' AND fig.object_id = %s', (ref_id, figure_id))
            res = tmp_cur.fetchone()
            if res[0] is True:
                qsr_graph.add_edge(figure_id, ref_id, QSR='above')
                isAbove = True
            if res[1] is True: qsr_graph.add_edge(figure_id, ref_id, QSR='below')
        except:
            print("Query too large, server problem raised")
            print(ref_id + " " + figure_id)
            return qsr_graph

        try:
            tmp_cur.execute('SELECT ST_3DIntersects(fig.bbox,reff.lefthsproj),ST_3DIntersects(fig.bbox,reff.righthsproj)'
                            ' from semantic_map as reff, semantic_map as fig'
                            ' WHERE reff.object_id = %s' \
                            ' AND fig.object_id = %s', (ref_id, figure_id))
            res = tmp_cur.fetchone()
            if res[0] is True:
                qsr_graph.add_edge(figure_id, ref_id, QSR='leftOf')
                qsr_graph.add_edge(figure_id, ref_id, QSR='beside')

            if res[1] is True:
                qsr_graph.add_edge(figure_id, ref_id, QSR='rightOf')
                qsr_graph.add_edge(figure_id, ref_id, QSR='beside')

        except:
            print("Query too large, server problem raised")
            print(ref_id + " " + figure_id)
            return qsr_graph
        try:
            tmp_cur.execute('SELECT ST_3DIntersects(fig.bbox,reff.fronthsproj), ST_3DIntersects(fig.bbox,reff.backhsproj)'
                            ' from semantic_map as reff, semantic_map as fig'
                            ' WHERE reff.object_id = %s' \
                            ' AND fig.object_id = %s', (ref_id, figure_id))
            res = tmp_cur.fetchone()
            if res[0] is True: qsr_graph.add_edge(figure_id, ref_id, QSR='inFrontOf')

            if res[1] is True: qsr_graph.add_edge(figure_id, ref_id, QSR='behind')

        except:
            print("Query too large, server problem raised")
            print(ref_id + " " + figure_id)
            return qsr_graph

        #Infer more complex QSR by combining the base ones (e.g., touch and above --> on top of)
        if isAbove is True and touches is True: qsr_graph.add_edge(figure_id, ref_id, QSR='onTopOf')
        # to test the other cases of ON (affixedOn and leanOn) we need all base QSRs with other objects gathered first

        if not qsr_graph.has_edge(figure_id, ref_id): # If not special rel, by default/definition, they are neighbours
            qsr_graph.add_edge(figure_id, ref_id, QSR='near')

    return qsr_graph

def extract_surface_QSR(session, obj_id, qsr_graph, fht=0.15, wht=0.2):
    """Extract QSRs through PostGIS
    between current object and surfaces marked as wall/floor
    fht: threshold to find objects that are at floor height, i.e., min Z coordinate = 0
    wht: for near wall surfaces - e.g., by default 20 cm
    motivation for threshold: granularity of map/GUI for wall annotation requires tolerance
    + account that walls are modelled 2D surfaces without depth in the GIS db, i.e., needs higher value than fht

    Expects a table of precomputed values formatted as per ./data/space_prep.py
    """
    tmp_conn, tmp_cur = session
    # Use postGIS for deriving truth values of base operators
    tmp_cur.execute("""SELECT o_zmin
                    FROM objects_precalc
                    WHERE object_id = %s
                    """,(obj_id,))
    # Unpack results and infer QSR predicates
    res = tmp_cur.fetchone()[0]
    if res <= fht:
        qsr_graph.add_edge(obj_id, 'floor', QSR='touches')
        #2020-05-15-11-24-12_379522_poly9 qsr_graph.add_edge(obj_id, 'floor', QSR='above')
        qsr_graph.add_edge(obj_id, 'floor', QSR='onTopOf')
        #also add relations in opposite direction #this is only to infer special ON cases later
        qsr_graph.add_edge('floor', obj_id, QSR='touches')
        qsr_graph.add_edge('floor', obj_id, QSR='below')

    tmp_cur.execute("""SELECT ow_distance
                       FROM objects_precalc
                       WHERE object_id = %s
                        """, (obj_id,))
    res = [r[0] for r in tmp_cur.fetchall()] #[0]
    #find the min distance to walls
    ws = min(res)
    if ws <= wht:
        qsr_graph.add_edge(obj_id, 'wall', QSR='touches')
        #qsr_graph.add_edge('wall', obj_id, QSR='touches') #also add relation in opposite direction
    return qsr_graph

def infer_special_ON(local_graph):
    """Iterates only over QSRs in current image
    but propagates new QSRs found to global graph
    expects label mapping in human-readable form as input"""
    for node1 in local_graph.nodes():
        # if obj1 touches or is touched by obj2
        #cobj = lmapping[node1]
        t = [(f,ref,r) for f,ref,r in local_graph.out_edges(node1, data=True) if r['QSR'] =='touches']
        is_t = [(f,ref,r) for f,ref,r in local_graph.in_edges(node1, data=True) if r['QSR'] =='touches']
        is_a = [f for f,_,r in local_graph.in_edges(node1, data=True) if r['QSR']=='below'] #edges where obj1 is reference and figure objects are below it
        t_rs = list(set([ref for _,ref,_ in t]))
        if len(t)==0 and len(is_t)==0: continue #skip
        elif len(t)==1 and len(is_t)==0 or (len(t_rs) ==1 and t_rs=='wall'): #exactly one touch relation or touching only walls
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
                if len(l)==0 and len(lb)==0 and len(others_below)>0: #and there is at least an o3 different from o2 which is below o1
                    local_graph.add_edge(node1, node2, QSR='leansOn') # then o1 leans on o2
    return local_graph

def extract_size(session, obj_id):
    # Retrieve object dimensions from spatial db
    # Note: implies that the ./utils/size_prep.py script has already be run first
    tmp_conn, tmp_cur = session
    tmp_cur.execute("""SELECT d1,d2,d3 FROM semantic_map
                        WHERE object_id = %s""", (obj_id,))
    res = tmp_cur.fetchone()
    return res[0], res[1], res[2]  #list of (d1,d2,d3)