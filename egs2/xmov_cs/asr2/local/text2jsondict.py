#!/usr/bin/env python3
import sentencepiece as spm
import json
import sys

input_text = sys.argv[1]
output_dict = sys.argv[2]
bpe_model = sys.argv[3]

sp = spm.SentencePieceProcessor()
sp.Load(bpe_model)

dictionary = dict()
with open(input_text, 'r', encoding='utf-8') as fin:
    for line in fin:
        line = line.strip()
        text = ' '.join(line.split(' ')[1:])
        pieces = sp.EncodeAsPieces(text)
        for p in pieces:
            if not p in dictionary:
                dictionary.update({p: 0})
            dictionary[p] += 1

json_str = json.dumps(dictionary, ensure_ascii=False, indent=4)
with open(output_dict, 'w', encoding='utf-8') as json_file:
    json_file.write(json_str)