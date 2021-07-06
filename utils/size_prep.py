import open3d as o3d
import csv
import numpy as np
import psycopg2
from psycopg2 import Error
import keyring # used for more secure pw storage
import sys
import os
import time
import statistics

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

def load_camera_intrinsics_txt(path_to_intr):
    """
    Expects 3x3 intrinsics matrix as tab-separated txt
    """
    intrinsics=[]
    with open(path_to_intr) as f:
        reader = csv.reader(f,delimiter='\t')
        for row in reader:
            if row==[]: continue
            for cell in row:
                if cell=='': continue
                try:
                    intrinsics.append(float(cell.split("  ")[1]))
                except IndexError:
                    try:
                        intrinsics.append(float(cell.split(" ")[1]))
                    except IndexError:
                        intrinsics.append(float(cell))
    return intrinsics

def pcl_remove_outliers(pcl, vx=5, std_r=2.):
    uni_down_pcd = pcl.uniform_down_sample(every_k_points=vx)
    neighbours = int(len(np.asarray(uni_down_pcd.points)))
    fpcl,_ = pcl.remove_statistical_outlier(nb_neighbors=neighbours, std_ratio=std_r)
    return fpcl

def estimate_dims(pcd,original_pcd):
    """
    Finds bounding solid and
    estimates obj dimensions from vertices
    """
    try:
        orthreedbox = pcd.get_oriented_bounding_box()
    except:
        print("Not enough points in 3D cluster, reduced to planar surface... reverting back to full pcd")
        #print(str(e))
        try:
            orthreedbox = original_pcd.get_oriented_bounding_box()
        except:
            return

    dims = orthreedbox.extent.tolist()
    nomin = dims.copy()
    nomin.remove(min(dims))
    """
    Hard to know a priori what is the w and what is the h
    But we can assume the depth will be always the min due to how data are captured
    """
    return (*nomin,min(dims))

def update_db(connection, cursor, filename, obj_dims):
    obj_id = filename.replace('depth', '')[:-4]
    if any(obj_dims) is None: # no size dims to use
        cursor.execute('UPDATE semantic_map SET d1= NULL, d2= NULL, d3=NULL '
                       ' WHERE object_id = %s',(obj_id,))
    else:
        d1,d2,d3 = obj_dims
        cursor.execute('UPDATE semantic_map SET d1= %s, d2= %s, d3=%s '
                   ' WHERE object_id = %s', (str(d1),str(d2),str(d3), obj_id))
    connection.commit()
    return connection, cursor

def main():
    scale = 1000.
    camintr = load_camera_intrinsics_txt('./data/KMi-set-2020-test2/'
                                         'camera-intrinsics.txt')
    camera = o3d.camera.PinholeCameraIntrinsic()
    camera.set_intrinsics(640, 480, camintr[0], camintr[4], camintr[2], camintr[5])

    depthps = []
    processingts = []
    for dirp, _, filen in os.walk('./data/KMi-set-2020-test2/test-imgs/'):
        if len(filen) > 0:  # skip folder level
            for fn in filen:
                if 'depth' in fn:
                    depthps.append((os.path.join(dirp,fn),fn))

    conn, cur = connect_DB(os.environ['USER'],'gis_database')
    for depthp,fn in depthps:
        start =time.time()
        depthimg = o3d.io.read_image(depthp)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depthimg, camera,depth_scale=scale, depth_trunc=scale)
        pcl_points = np.asarray(pcd.points).shape[0]
        if pcl_points <= 1:
            conn, cur = update_db(conn,cur, fn,(None,None,None))

        cluster_pcl = pcl_remove_outliers(pcd)
        try:
            d1, d2, d3 = estimate_dims(cluster_pcl, pcd)
            print("Estimated dims oriented %f x %f x %f m" % (d1, d2, d3))
        except TypeError:
            #not enough points, skip
            d1,d2,d3 = None,None,None
        conn, cur = update_db(conn,cur, fn,(d1,d2,d3))
        processingts.append(float(time.time() - start))

    disconnect_DB(conn, cur)
    print("Object sizes added to semantic map")
    print("Mean processing time: %f" % float(statistics.mean(processingts)))
    print("St dev processing time: %f" % float(statistics.stdev(processingts)))

if __name__ == '__main__':
    sys.exit(main())