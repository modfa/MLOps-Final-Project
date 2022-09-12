import json
import uuid
from datetime import datetime
from time import sleep
import pandas as pd
from pyarrow import csv
import pyarrow as pa

import pyarrow.parquet as pq
import requests

df = csv.read_csv("cars_sampled.csv")
# table = pa.Table.from_pandas(df)
# table = pd.read_csv("cars_sampled.csv")
# table = df.values.tolist()

# table1 = pa.Table.from_pandas(df)
# pq.write_table(table1, 'cars_sampled.parquet')
tablee = pa.Table.from_pandas(df)
pq.write_table(tablee, 'file_name.parquet')
table1 = pq.read_table("file_name.parquet")
# table = pq.read_table("cars_sampled.parquet")
# data = table.to_pylist()
table = table1.to_pydict()
# table = list(table1)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


with open("target.csv", 'w') as f_target:
    for row in data:
        row['id'] = str(uuid.uuid4())
        price = row['price']
        if price != 0.0:
            f_target.write(f"{row['id']},{price}\n")
        resp = requests.post("http://127.0.0.1:9696/predict",
                             headers={"Content-Type": "application/json"},
                             data=json.dumps(row, cls=DateTimeEncoder)).json()
        print(f"prediction: {resp['price']}")
        sleep(1)
