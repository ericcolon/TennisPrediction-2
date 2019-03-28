import csv, pyodbc
import os
import subprocess
import sys

# Add the most current database version as a local path.

MDB = "/Users/aysekozlu/PyCharmProjects/TennisModel/OnCourt.mdb" # This is local. Change this to os.getpath

# Dump the schema for the DB
subprocess.call(["mdb-schema", MDB, "mysql"])

# Get the list of table names with "mdb-tables"
table_names = subprocess.Popen(["mdb-tables", "-1", MDB],
                               stdout=subprocess.PIPE).communicate()[0]
tables = table_names.splitlines()

print
"BEGIN;"  # start a transaction, speeds things up when importing
sys.stdout.flush()

# Dump each table as a CSV file using "mdb-export",
# converting " " in table names to "_" for the CSV filenames.
for table in tables:
    if table != '':
        subprocess.call(["mdb-export", "-I", "mysql", MDB, table])

print
"COMMIT;"  # end the transaction
sys.stdout.flush()
