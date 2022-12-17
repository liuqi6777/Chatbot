python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/queries \
  --language zh \
  --index index/queries \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw