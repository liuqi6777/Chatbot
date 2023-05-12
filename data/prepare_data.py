import json
import os

from tqdm import tqdm


if __name__ == '__main__':

    with open('data/lccc_large.jsonl', 'r', encoding='utf-8') as fin, open('data/single_turn.jsonl', 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            data = json.loads(line.rstrip())
            if len(data) == 2:
                data = [''.join(x.split()) for x in data]
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')

    os.mkdir('data/queries', exist_ok=True)
    with open('data/single_turn.jsonl', 'r', encoding='utf-8') as fin, open('data/queries/queries.jsonl', 'w', encoding='utf-8') as fout:
        for i, line in tqdm(enumerate(fin)):
            query = json.loads(line.rstrip())[0]
            fout.write(json.dumps({'id': str(i), 'contents': query}, ensure_ascii=False))
            fout.write('\n')
                    
    print('done')
