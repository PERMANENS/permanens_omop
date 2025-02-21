import psycopg2
import polars as pl
import argparse
import sys

DB_NAME = 'permanens_v6'
HOST = 'localhost'
PORT = 5432
USER = 'giacomo'

def main ():
    parser = argparse.ArgumentParser()
    parser.add_argument('-P', '--password', type=str, required=False, help='Password for database access (required)')
    
    # Parse the arguments
    args = parser.parse_args()  
    if args.password != None:
        sys.stdout.write('checking the integrity of local DB instance...\n')
        sys.stdout.flush()
    else:
        sys.stdout.write('Please intert the correct password to access the DB...\n')
        sys.stdout.flush()

    visit_tables=["person","condition_occurrence","concept","visit_occurrence","observation","death","drug_exposure"]
    visit_columns=["person_id","visit_occurrence_id", "observation_source_value","year_of_birth","concept_name","visit_concept_id","visit_start_date","visit_end_date"]
    miniPL_tables=["person","condition_occurrence","visit_occurrence","observation","death","drug_exposure","concept"]
    miniPL_columns=["person_id","condition_source_value","observation_source_value","concept_name","condition_start_date","cause_source_value","cause_concept_id"]

    password = args.password
    #Connect to the database
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=USER,
        password=password,
        host=HOST,
        port=PORT)

    with conn.cursor() as cur:
        cur.execute("SET SCHEMA 'cdm_permanens_trial_6';")
        conn.commit()

    query = "SELECT table_name AS table, column_name AS column FROM information_schema.columns WHERE table_schema NOT IN ('pg_catalog', 'information_schema');"
    DB_info = pl.read_database(query,conn)
    

    DB_table_list=[]
    DB_column_list=[]
    all_present=True
    for entry in DB_info.iter_rows():
        table = entry[0]
        column = entry[1]
        if table not in DB_table_list: DB_table_list.append(table)
        if column not in DB_column_list: DB_column_list.append(column)

    for visit_table in visit_tables:
        if visit_table not in DB_table_list:
            print(f"{visit_table} not found in the DB")
            all_present=False

    for miniPL_table in visit_tables:
        if miniPL_table not in DB_table_list:
            print(f"{miniPL_table} not found in the DB") 
            all_present=False

    for visit_column in visit_columns:
        if visit_column not in DB_column_list:
            print(f"{visit_column} not found in the DB")
            all_present=False

    for miniPL_column in miniPL_columns:
        if miniPL_column not in DB_column_list:
            print(f"{miniPL_column} not found in the DB")
            all_present=False

    if all_present: print("All the tables and columns are present in the DB...")

if __name__ == '__main__':
    main()
