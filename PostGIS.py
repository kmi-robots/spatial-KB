"""
Interface methods for spatial database
"""
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
        "predicate_synset varchar,"
        "subject_polygon geometry,"
        "object_polygon geometry,"
        "ST_intersects boolean,"
        "ST_crosses boolean,"
        "ST_overlap boolean,"
        "ST_touches boolean,"
        "ST_within boolean,"
        "ST_contains boolean);")
    print("DB table VG_RELATIONS created or exists already")

def add_VG_row(cursor,qargs):
    rel_id, pred, pred_synset, sub_coords, obj_coords = qargs
    scoords_= [str(x)+" "+str(y) for x,y in sub_coords]
    ocoords_= [str(x)+" "+str(y) for x,y in obj_coords]
    spoly_str = "POLYGON(("+",".join(t for t in scoords_)+"))"
    opoly_str = "POLYGON(("+",".join(t for t in ocoords_)+"))"

    cursor.execute("""INSERT INTO VG_RELATIONS(relation_id, predicate_name, predicate_synset,subject_polygon,object_polygon)
                   VALUES (%s,%s,%s,ST_GeomFromText(%s),ST_GeomFromText(%s));
                   """,(rel_id,pred,str(pred_synset),spoly_str,opoly_str))

    print("Relation datapoint %s added to VG_RELATIONS" % rel_id )

