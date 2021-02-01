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