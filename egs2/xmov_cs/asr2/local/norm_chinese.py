#!/usr/bin/env python3

from harvesttext import HarvestText
from cn_tn import TextNorm
ht = HarvestText()
tn = TextNorm(
        to_banjiao = True,
        to_upper = False,
        to_lower = True,
        remove_fillers = False,
        remove_erhua = False,
        check_chars = False,
        remove_space = False,
    )

import sys
input = sys.argv[1]
output = sys.argv[2]

fout = open(output, 'w', encoding='utf-8')
with open(input, 'r', encoding='utf-8') as fin:
    for line in fin:
        line = line.strip()
        line = line.split('\t')
        #sents = ht.cut_sentences(line, deduplicate=True)
        name = line[0]
        if len(line) > 1 :
            text = line[1]
            #for s in sents:
            clean_text = ht.clean_text(text, norm_html=True, norm_url=True, remove_puncts=True)
            text_tn = tn(clean_text)
            fout.write(name + ' ' + text_tn+'\n')
        else:
            fout.write(name + ' ' + '\n')
        fout.flush()
fout.close()
