#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import os
import psycopg2
from decimal import Decimal
from datetime import datetime



class PostgresUtils:
    def __init__(self, host, port, database, user, pwd):
        self.host = host
        self.port = port 
        self.database = database
        self.user = user
        self.pwd = pwd

    def create_connection(self): 
        try: 
            return psycopg2.connect(database=self.database, user=self.user, password=self.pwd)
        except Exception as e: 
            print(f"连接失败: {e}")
            return None
    
    def execute_sql(self, conn, sql_statement):
        try: 
            with conn.cursor() as cur: 
                cur.execute(sql_statement)
                conn.commit()
                columns = [desc.name for desc in cur.description]
                exe_res = cur.fetchall()
                result = []

                for line in exe_res:
                    line = list(line)
                    for i, val in enumerate(line):
                        if isinstance(val, Decimal):
                            line[i] = float(val)
                        if isinstance(val, datetime):
                            line[i] = str(val)
                    result.append(dict(zip(columns, line)))
                return result
        except Exception as e: 
            conn.rollback()
            print(f"sql执行失败: {e}")

    def close(self, conn): 
        if conn: 
            conn.close()

    
    def get_schema(self, table_schema, table_name):
        try: 
            conn = self.create_connection()
            sql = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = '{table_schema}';"
            return self.execute_sql(conn, sql)
        except Exception as e: 
            conn.rollback()
            print(f"获取表结构失败: {e}")


if __name__ == "__main__":
    # pass
    pg_utils = PostgresUtils("localhost", 5432, "postgres", "monacui", "1234")
    conn = pg_utils.create_connection()
    pg_utils.execute_sql(conn, "SELECT * FROM llm.tbl_super_store LIMIT 1;")
    print(pg_utils.get_schema("llm", "tbl_super_store"))
    pg_utils.close(conn)
