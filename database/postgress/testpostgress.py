import psycopg2

# conn = psycopg2.connect("dbname=suppliers user=raghav password=Bardock123$")

# conn = psycopg2.connect(host="localhost",port="5434",database="suppliers",user="raghav",password="Bardock123$")
conn = psycopg2.connect(host="localhost",port="5434",database="pyapp_stock_data", user="postgres")
#Creating a cursor object using the cursor() method
cursor = conn.cursor()

#Executing an MYSQL function using the execute() method
cursor.execute("select version()")

# Fetch a single row using fetchone() method.
data = cursor.fetchone()
print("Connection established to: ",data)

#Closing the connection
conn.close()