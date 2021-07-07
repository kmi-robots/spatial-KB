""" Spatial DB preparation
    semantic_map table will be shared as a data dump,
    as well as walls table, this script implies both are present and precalc the rest
    """
import psycopg2
from psycopg2 import Error
import keyring
import sys
import os

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

def main():

    conn, cur = connect_DB(os.environ['USER'],'gis_database')
    # create index on semantic map ids first for faster access later
    cur.execute("""
            CREATE INDEX objid_index
            on semantic_map 
            USING hash (object_id)
            """)
    conn.commit()
    disconnect_DB(conn, cur)

    conn, cur = connect_DB(os.environ['USER'], 'gis_database')
    #create table with precomputed distance to walls and object zmin

    cur.execute("""CREATE TABLE IF NOT EXISTS objects_precalc(
                    record_id serial primary key,
                    object_id varchar NOT NULL,
                    wall_id varchar NOT NULL,
                    bbox geometry,
                    surface geometry,
                    ow_distance float,
                    o_zmin float)
                """ )
    conn.commit()
    #insert data into table
    cur.execute("""
    INSERT INTO objects_precalc(object_id, bbox,o_zmin, wall_id, surface,ow_distance)
    SELECT sm.object_id, sm.bbox, ST_Zmin(sm.bbox), w.id, w.surface, 
	ST_3DDistance(sm.bbox,w.surface)
	FROM semantic_map as sm, walls as w
	where sm.object_polyhedral_surface is not null
    """)
    conn.commit()
    disconnect_DB(conn, cur)

    conn, cur = connect_DB(os.environ['USER'], 'gis_database')
    #add index on both object_id and wall_id
    cur.execute("""
    CREATE INDEX obj_id_index
    on objects_precalc 
    USING hash
    (object_id)
    """)
    conn.commit()
    disconnect_DB(conn, cur)

    conn, cur = connect_DB(os.environ['USER'], 'gis_database')
    cur.execute("""
        CREATE INDEX wall_id_index
        on objects_precalc 
        USING hash
        (wall_id)
        """)
    conn.commit()
    disconnect_DB(conn,cur)
    return 0

if __name__ == '__main__':
    sys.exit(main())