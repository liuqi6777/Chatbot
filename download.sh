#!/bin/bash

# cd scratch place
cd data/

# Download zip dataset from Google Drive
filename='raw_chat_corpus.zip'
fileid='1So-m83NdUHexfjJ912rQ4GItdLvnmJMD0B81rNlvomiwed0V1YUxQdC1uOTg'
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/u/0/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

cd