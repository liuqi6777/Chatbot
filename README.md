# Chatbot

## introduction

## How to use it

### Prepare data and build index

```bash
# download datasets
sh data/download.sh

# prepare single turn dialog dataset
python prepare_data.py

# build inverted index for query
sh indexing/build_query_index.sh

# build a sql table which allow us to select response given query's id
python indexing/build_response_table.py
```

The built index is under `./index` folder. 

### Run the project

After you finished above preparations, you can run the project directly:

```bash
python app.py
```

If you want to change the config (e.g. only use retrieval module but not generating module), please refer to `config.py`.

