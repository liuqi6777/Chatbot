import ujson
import sqlite3
from tqdm import tqdm


con = sqlite3.connect('index/responses.sqlite')
cur = con.cursor()
cur.execute(
    """
    create table responses (
        query_id INT PRIMARY KEY,
        response VARCHAR(100)
    )
    """
)

with open('data/single_turn.jsonl', 'r', encoding='utf-8') as f:
    for i, line in tqdm(enumerate(f)):
        data = ujson.loads(line.rstrip())
        query, response = data[0], data[1]
        cur.execute("insert into responses values (?, ?)", (i, response))


con.commit()
cur.close()
con.close()
