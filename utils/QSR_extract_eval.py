from PostGIS import *
import pandas as pd
import sys, os
import networkx as nx
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.graphs import plot_graph

def populate_with_boxes(connection,cursor, sf=1.2):
    """Creation of min oriented bounding box, contextualised bounding box
    and six halfspaces, for each object/spatialRegion
    """

    # for all objects in spatial DB
    # except crops without depth data associated , i.e., if obj polyhedral surface or projection2d is null
    # Find min oriented 3D box bounding of polyhedral surface
    cursor.execute('SELECT object_id, ST_OrientedEnvelope(projection_2d), ST_ZMin(object_polyhedral_surface),'
                   ' ST_ZMax(object_polyhedral_surface) FROM semantic_map '
                         'where object_polyhedral_surface is not null;')
    # Also filter out blacklisted objects that were not run with size reasoning
    blacklist = ['246928_6', '655068_5', '655068_6']
    query_res = [(str(r[0]), str(r[1]), float(r[2]), float(r[3])) for r in cursor.fetchall() \
                 if '_'.join(str(r[0]).split('_')[1:]) not in blacklist]

    for id_, envelope, zmin, zmax in query_res:

        #if id_=='2020-05-15-11-12-27_125515_poly0':
        #    print("Hold")
        height = zmax - zmin
        # ST_Translate is needed here because the oriented envelope is projected on XY so we also need to translate
        # everything up by the original height after extruding in this case
        up1_mask = 'UPDATE semantic_map SET bbox = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s)' \
                   ' WHERE object_id = %s;'
        cursor.execute(up1_mask, (envelope, str(height), zmin, id_))
        connection.commit()

        #Derive CBB
        #minimum of the angles between:
        #1) line connecting robot position and object centroid
        #2) line extending each of the edges of the oriented base of the object


        cursor.execute('SELECT ST_Angle(ST_MakeLine(robot_position, ST_Centroid(ST_OrientedEnvelope(projection_2d))), ' \
            'ST_MakeLine(ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(projection_2d)), 1), ' \
            'ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(projection_2d)), 2))), ' \
            'ST_Angle(ST_MakeLine(robot_position, ST_Centroid(ST_OrientedEnvelope(projection_2d))), ' \
            'ST_MakeLine(ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(projection_2d)), 2), ' \
            'ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(projection_2d)), 3))), ' \
            'ST_Angle(ST_MakeLine(robot_position, ST_Centroid(ST_OrientedEnvelope(projection_2d))), ' \
            'ST_MakeLine(ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(projection_2d)), 3), ' \
            'ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(projection_2d)), 4))), ' \
            'ST_Angle(ST_MakeLine(robot_position, ST_Centroid(ST_OrientedEnvelope(projection_2d))), ' \
            'ST_MakeLine(ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(projection_2d)), 4), ' \
            'ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(projection_2d)), 1)))'\
            ' FROM semantic_map ' \
            ' WHERE object_id = \'' + id_ + '\';')


        # Derive CBB
        # cursor.execute(
        #      'SELECT ST_Angle(ST_MakeLine(robot_position, ST_Centroid(ST_OrientedEnvelope(projection_2d))), '
        #      'ST_MakeLine(ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(projection_2d)), 1), '
        #      'ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(projection_2d)), 2))) '
        #      'FROM semantic_map '
        #      'WHERE object_id = \'' + id_ + '\';')

        #Angle between:
        # 1) line connecting robot position and object centroid
        # 2) line connecting robot position with closest point on 2D envelope of object
        # cursor.execute(
        #     'SELECT ST_Angle(ST_MakeLine(robot_position, ST_Centroid(ST_OrientedEnvelope(projection_2d))), '
        #     'ST_ShortestLine(robot_position,ST_OrientedEnvelope(projection_2d))) '
        #     'FROM semantic_map '
        #     'WHERE object_id = \'' + id_ + '\';')

        connection.commit()

        angles = list(cursor.fetchone())
        angle = min(angles)


        up2_mask = 'UPDATE semantic_map SET cbb = ST_Rotate(bbox, %s,' \
                   ' ST_Centroid(ST_OrientedEnvelope(projection_2d)))' \
                   ' WHERE object_id = %s;'
        cursor.execute(up2_mask, (str(angle), id_))
        connection.commit()

        # Derive the six halfspaces, based on scaling factor sf

        # top and bottom ones based on MinOriented, i.e., extruded again from oriented envelope
        up_topbtm = 'UPDATE semantic_map SET  tophsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s),'\
                    ' bottomhsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s)'\
                   ' WHERE object_id = %s;'
        cursor.execute(up_topbtm, (envelope, str(height*sf), zmax, envelope, str(height * sf), zmin-height*sf, id_))

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

        cursor.execute(q_hs, (str(sf),str(sf), id_))
        q_res = cursor.fetchall() # for each object, 2 rows by 3 colums (i.e., 4 halfspaces + base of cbb repeated twice)

        # Interpret what is L/R/front/back among those boxes
        all_dis =[]
        all_angles = []
        #allfbs = [q[1] for q in q_res]
        #alllrs = [q[0] for q in q_res]
        all_hss=[]
        for q in q_res:
            all_hss.append(q[0])
            all_hss.append(q[1])
        #all_hss = [q[0],q[1] for q in q_res] + [q[1] for q in q_res]
        for lr, fb, base,rp in q_res:

            qdis = 'SELECT St_Distance(%s, St_Centroid(St_GeomFromEWKT(%s)))'
            cursor.execute(qdis, (rp, lr,))
            qdsir = cursor.fetchone()
            qdisr2 = qdsir[0]
            all_dis.append(qdisr2)

            # Distance between robot position and hs centroid
            cursor.execute(qdis, (rp,fb,))
            qdisr = cursor.fetchone()[0]
            all_dis.append(qdisr)

            # angle between robot position and base centroid (St_Angle is computed clockwise)
            qang = 'SELECT St_Angle(St_MakeLine(%s,St_Centroid(%s)), St_MakeLine(%s '\
                        ',St_Centroid(St_GeomFromEWKT(%s))))'
            cursor.execute(qang, (rp,base,rp,lr))
            qangr = cursor.fetchone()[0]
            all_angles.append(qangr)

            cursor.execute(qang, (rp, base, rp, fb))
            qangr = cursor.fetchone()[0]
            all_angles.append(qangr)

        #front is the nearest one to robot position
        front_idx = all_dis.index(min(all_dis))
        fronths = all_hss[front_idx]
        #Is the index found for fronths odd or even? Take other hs along same axis as back hs
        if front_idx % 2 == 0:
            back_idx = [indd for indd in range(4) if indd %2==0 and indd!=front_idx][0] #other even index that is not the fron one
        else:
            back_idx = [indd for indd in range(4) if indd %2!=0 and indd!=front_idx][0] #other even index that is not the fron one

        backhs = all_hss[back_idx]
        #fronths = q_res[all_dis.index(min(all_dis))][1]
        #Left one has the biggest angle with robot position and base centroid

        #remove front and back from angle list as the closest one may be close to 360 degrees, i.e., aligned with robot's position
        old_ids = [az for az, a in enumerate(all_angles) if az not in [front_idx, back_idx]]
        all_angles = [a for az, a in enumerate(all_angles) if az not in [front_idx,back_idx]]

        left_idx = all_angles.index(max(all_angles))
        left_idx = old_ids[left_idx]
        lefths = all_hss[left_idx]

        #right hs then is the only remaining index
        right_idx = [indd for indd in range(4) if indd not in [front_idx,back_idx,left_idx]][0]
        righths = all_hss[right_idx]
        #backhs = [fb for fb in allfbs if fb!= fronths][0] # the one record which is not the front one will be the back one
        #righths = [lr for lr in alllrs if lr!= lefths][0] # similarly for L/R
        # Extrude + Translate 3D & update table with halfspace columns
        up_others = 'UPDATE semantic_map SET  lefthsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s),' \
                    ' righthsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s),' \
                    ' fronthsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s),' \
                    ' backhsproj = ST_Translate(ST_Extrude(%s, 0, 0, %s), 0, 0, %s)' \
                    ' WHERE object_id = %s;'
        cursor.execute(up_others,
                                (lefths, str(height), zmin, righths, str(height), zmin,
                                 fronths,str(height), zmin, backhs,str(height), zmin, id_))
        connection.commit()


def format_ground_truth(dataf,ldict):

    for r_ in ldict.keys():
        rel_col = dataf[r_].dropna()
        for ridx, rval in rel_col.iteritems():

            row = dataf.iloc[[ridx]]  # extract row for value in column
            obj1 = row['image_id'].tolist()[0] + '_' + row['region_num'].tolist()[0]

            if rval =='floor' or rval=='wall':
                obj2 = rval
                ldict[r_].append((obj1, obj2))
            else:
                refs = rval.split(',')
                if len(refs)>1:
                    for ref in refs:
                        obj2 = row['image_id'].tolist()[0] + '_' + ref
                        if ref == 'floor' or ref == 'wall':
                            obj2 = ref
                        ldict[r_].append((obj1, obj2))
                else:
                    obj2 = row['image_id'].tolist()[0] + '_' + rval
                    ldict[r_].append((obj1, obj2))

    return {r_.lower():v_ for r_,v_ in ldict.items()}


def eval_QSR(relations,gt_dict):

    for k,nodelist in gt_dict.items():
        print("----Evaluating relation %s----" % k)
        extr_rels = relations[k]
        gtlabs = []
        preds = []
        for o1,o2 in nodelist:
            gtlabs.append(1)
            if (o1, o2) in extr_rels:
                preds.append(1)
            else:
                print("Missing %s %s %s" % (o1,k,o2))
                preds.append(0)

        acc = accuracy_score(gtlabs,preds) #float(len([p for p in preds if p==1])/len(gtlabs))
        p,r,f1,supp = precision_recall_fscore_support(gtlabs,preds) #p,r,f1,supp = precision_recall_fscore_support(gtlabs,preds,output_dict=True)
        print("Accuracy: %f" % acc)
        print("Precision: %f, Recall: %f, F1: %f" % (p[1],r[1],f1[1]))
        print("Suppport: %i" % int(supp[1]))

def main():

    path_to_annotations = './data/annotated_QSR_subset.csv'
    df = pd.read_csv(path_to_annotations)
    imgid_list = list(set(df['image_id'].tolist()))

    target_rels = [r for r in list(df.columns.values) if r not in ['image_id','region_num','gt_label']]
    # target_nums = list(range(len(target_rels)))
    gtruth = {r:[] for r in target_rels}
    extracted={r.lower():[] for r in target_rels}

    #Prep ground truth annotations
    gtruth = format_ground_truth(df,gtruth)

    #Extract QSR from data
    # populate_with_boxes(*session) #to exec only on first run

    # print("DB updated with boxes")
    for imgid in imgid_list: #['2020-05-15-11-03-03_130652']: #imgid_list
        session = connect_DB(os.environ['USER'], 'gis_database')
        QSRs = nx.MultiDiGraph()
        ids_ord = retrieve_ids_ord(session, imgid) # find all spatial regions at timestamp in db and order by volume desc

        QSRs.add_nodes_from(ids_ord.keys())
        for i, o_id in enumerate(ids_ord.keys()):
            figure_objs = find_neighbours(session, o_id, ids_ord)
            if len(figure_objs) > 0:
                QSRs = extract_QSR(session, o_id, figure_objs, QSRs) # relations between objects
            QSRs = extract_surface_QSR(session, o_id, QSRs,fht=0.15, wht=0.259) # relations with walls and floor
        # after all reference objects have been examined
        # derive special cases of ON
        QSRs = infer_special_ON(QSRs)

        #When on top of, remove above -- if affixed on, consider also InFrontOf,L,R or behind

        # plot_graph(QSRs)
        QSRs_fil = QSRs.copy() # remove redundant rels
        # print(QSRs_fil.edges(data=True,keys=True))
        for (n1, n2, k, d) in QSRs.edges(data=True,keys=True):
            rel_name = d['QSR']
            print(n1+' '+rel_name+' '+n2)
            rem =False
            if rel_name =='touches' or rel_name=='beside' or rel_name=='near':
                QSRs_fil.remove_edge(n1,n2,k)
                rem=True
            if n1 =='wall' or n1=='floor' and not rem:
                QSRs_fil.remove_edge(n1,n2, k) #and not already removed above
                rem=True
            if not rem:
                extracted[rel_name.lower()].append((n1,n2)) #add to predictions
        disconnect_DB(*session)
        print("-------------------------")

    eval_QSR(extracted,gtruth)


if __name__ == '__main__':
    sys.exit(main())