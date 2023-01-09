import json

from tqdm import tqdm


if __name__ == '__main__':
    reader = open('data/lccc_large.jsonl', 'r', encoding='utf-8')
    writer = open('data/single_turn.jsonl', 'w', encoding='utf-8')
    for line in tqdm(reader):
        data = json.loads(line.rstrip())
        if len(data) == 2:
            data = [''.join(x.split()) for x in data]
            writer.write(json.dumps(data, ensure_ascii=False) + '\n')
    reader.close()
    writer.close()

    with open('data/single_turn.jsonl', 'r', encoding='utf-8') as fin:
        with open('data/queries.jsonl', 'w', encoding='utf-8') as fout:
            for i, line in tqdm(enumerate(fin)):
                query = json.loads(line.rstrip())[0]
                fout.write(json.dumps({'id': str(i), 'contents': query}, ensure_ascii=False))
                fout.write('\n')
                    
    print('done')
