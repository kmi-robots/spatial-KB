"""Create Web viz of 3D objects in postgis DB
Example below with map walls
and subset of objects
"""

import os
import sys
import psycopg2
from psycopg2 import Error
import keyring
from bs4 import BeautifulSoup


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

def generate_html_viz(cursor,tgtp, timestamp=None):
    """dbobj.cursor.execute('SELECT object_id, ST_AsX3D(bbox), ST_AsX3D(cbb), ST_AsX3D(tophsproj),'
                                'ST_AsX3D(bottomhsproj),ST_AsX3D(lefthsproj),ST_AsX3D(righthsproj),'
                                'ST_AsX3D(fronthsproj),ST_AsX3D(backhsproj), ST_X(robot_position),'
                                'ST_Y(robot_position), ST_Z(robot_position), ST_AsX3D(object_polyhedral_surface)  '
                                'FROM semantic_map '
                                'WHERE object_id LIKE %s;',(timestamp+'%',))
    qres =dbobj.cursor.fetchall()"""

    cursor.execute('SELECT ST_AsX3D(surface) from walls')
    qwalls = cursor.fetchall()

    # tgtp= os.path.join(os.environ['HOME'], 'walldumpviz.html')
    with open(tgtp, 'w') as outd:
        # rpos = " ".join([str(coord) for coord in qres[-2][-3:]]) #robot coords for given timestamp/image
        # Write html heading
        outd.write('<!DOCTYPE html><html><head><meta encoding="utf-8"><script src="http://www.x3dom.org/download/dev/x3dom.js"></script><link rel="stylesheet" '
            'href="http://www.x3dom.org/download/dev/x3dom.css"></head><body>\
                    <x3d xmlns="http://www.x3dom.org/x3dom" showStat="false" showLog="false" x="0px" y="0px" width="1920px" height="1080px"><scene>\
                    <viewpoint position="0 0 10"></viewpoint>' + '\n' ) # sample viewpoint was 0 0 10
        #robot viewpoint (in gray)
        # outd.write(('<Transform translation="%s">\
        # <Shape><Sphere radius=".02"/><Appearance USE="DARK_GRAY"/></Shape></Transform> ' + '\n')% rpos)

        #0,0,0 origin of postgis coord system
        #outd.write('<Transform translation="0 0 0">\
        #        <Shape><Sphere radius="1.0"/><Appearance USE="DARK_GRAY"/></Shape></Transform> ' + '\n')

        """for i,r in enumerate(qres): 
            # bbox is black (default)
            outd.write('<shape><appearance><material></material></appearance>' + '\n')
            outd.write(r[1].replace('FaceSet','LineSet') + '\n')
            outd.write('</shape>' + '\n')

            if i==0:#r[0]=='2020-05-15-11-13-38_805787_poly1': #draw all boxes/hs only for one for readability
                # CBBs are blue
                outd.write('<shape><appearance><material emissiveColor="0. 0. 1.0"></material></appearance>' + '\n')
                outd.write(r[2].replace('FaceSet','LineSet') + '\n')
                outd.write('</shape>' + '\n')
                #     #tophs are red
                #     #outd.write('<shape><appearance><material emissiveColor="1.0 0. 0.0"></material></appearance>' + '\n')
                #     #outd.write(r[3].replace('FaceSet','LineSet') + '\n')
                #     #outd.write('</shape>' + '\n')
                #     #btmhs green
                #     #outd.write('<shape><appearance><material emissiveColor="0. 1.0 0."></material></appearance>' + '\n')
                #     #outd.write(r[4].replace('FaceSet', 'LineSet') + '\n')
                #     #outd.write('</shape>' + '\n')
                # lefths are red
                outd.write('<shape><appearance><material emissiveColor="1.0 0. 0.0"></material></appearance>' + '\n')
                outd.write(r[5].replace('FaceSet', 'LineSet') + '\n')
                outd.write('</shape>' + '\n')
                # righths green
                outd.write('<shape><appearance><material emissiveColor="0. 1.0 0."></material></appearance>' + '\n')
                outd.write(r[6].replace('FaceSet', 'LineSet') + '\n')
                outd.write('</shape>' + '\n')
                #     # # fronths are red
                #     # outd.write('<shape><appearance><material emissiveColor="1.0 0. 0.0"></material></appearance>' + '\n')
                #     # outd.write(r[7].replace('FaceSet', 'LineSet') + '\n')
                #     # outd.write('</shape>' + '\n')
                #     # backhs green
                #     # outd.write('<shape><appearance><material emissiveColor="0. 1.0 0."></material></appearance>' + '\n')
                #     # outd.write(r[8].replace('FaceSet', 'LineSet') + '\n')
                #     # outd.write('</shape>' + '\n')
                #show convex hull
                outd.write('<shape><appearance><material emissiveColor="1.0 0. 0.0"></material></appearance>' + '\n')
                outd.write(r[-1] + '\n')
                outd.write('</shape>' + '\n')"""
        #draw walls
        for w in qwalls:
            outd.write('<shape><appearance><material></material></appearance>' + '\n')
            outd.write(w[0] +'\n')
            outd.write('</shape>' + '\n')

        # closing nodes
        outd.write('<shape><plane></plane></shape></scene></x3d></body></html>')
    print("HTML report of 3D boxes created under %s" % tgtp)

def generate_sql_from_viz(cursor, connection,filepath, wheight=4):
    """Opposite of function above, from html to postgis multipoint object"""
    with open(filepath) as htmlf:
        content = htmlf.read()
        soup = BeautifulSoup(content,features="html.parser")
        tags = soup.findAll("indexedfaceset")
        for n,wall_tag in enumerate(tags):
            points = wall_tag.contents[0]['point']
            single_coords = points.split(" ")
            x_ = single_coords[::3]
            xm, xm_ = x_[0], x_[2]
            y_ = single_coords[1::3]
            ym, ym_ = y_[0], y_[2]
            """z_ = single_coords[2::3]
            multipoint_string = 'MULTIPOINT('
            for i, (x,y,z) in enumerate(list(zip(x_,y_,z_))):
                if i< len(x_)-1:
                    multipoint_string += x+' '+y+' '+z+','
                else: # no comma and close parenthesis
                    multipoint_string += x + ' ' + y + ' ' + z + ')'
            cursor.execute('INSERT INTO TEST_WALLS(id,surface)'
                           'VALUES(%s, ST_GeomFromEWKT(%s))', (str(n),multipoint_string))
            """
            #insert into test walls table in db
            query_mask = "INSERT INTO walls(surface) " \
                         "VALUES(ST_Extrude(ST_GeomFromText('LINESTRING({} {}, {} {})'),0,0,{}))"
            query = query_mask.format(xm, ym, xm_, ym_, wheight)
            cursor.execute(query)

            continue

    connection.commit() #commit changes to db
    return


def main():
    conn, cur = connect_DB(os.environ['USER'], 'gis_database')
    output_p = os.path.join(os.environ['HOME'], 'hsdump_filtered.html')
    #generate_html_viz(cur,output_p)
    generate_sql_from_viz(cur,conn, output_p)
    disconnect_DB(conn, cur)
    return

if __name__ == '__main__':
    sys.exit(main())
