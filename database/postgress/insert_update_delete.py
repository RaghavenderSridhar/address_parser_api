import psycopg2
from psycopg2 import Error

try:
    # Connect to an existing database
    connection = psycopg2.connect(user="raghav",
                                  password="Bardock123$",
                                  host="127.0.0.1",
                                  port="5434",
                                  database="pyapp_stock_data")

    # Create a cursor to perform database operations
    cursor = connection.cursor()
    insert_query = """ INSERT INTO mobile (ID, MODEL, PRICE) VALUES (1, 'Iphone12', 1100)"""
    cursor.execute(insert_query)
    insert_query = """ INSERT INTO mobile (ID, MODEL, PRICE) VALUES (2, 'samsung', 5000)"""
    cursor.execute(insert_query)
    connection.commit()
    print("1 Record inserted successfully")
    # Fetch result
    cursor.execute("SELECT * from mobile")
    record = cursor.fetchall()
    print("Result ", record)

    # Executing a SQL query to update table
    update_query = """Update mobile set price = 1500 where id = 1"""
    cursor.execute(update_query)
    connection.commit()
    count = cursor.rowcount
    print(count, "Record updated successfully ")
    # Fetch result
    cursor.execute("SELECT * from mobile")
    print("Result ", cursor.fetchall())

    # Executing a SQL query to delete table
    delete_query = """Delete from mobile where id = 1"""
    cursor.execute(delete_query)
    connection.commit()
    count = cursor.rowcount
    print(count, "Record deleted successfully ")
    # Fetch result
    cursor.execute("SELECT * from mobile")
    print("Result ", cursor.fetchall())


except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)
finally:
    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")
