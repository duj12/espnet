#!/usr/bin/env python3
import sys

#subset = 'train'

src_scp = sys.argv[1] #f'dump/raw/{subset}/wav.scp'
tgt_scp = sys.argv[2] #f'data/{subset}/wav.scp'

src = {}
with open(src_scp) as f_s:
    for line in f_s:
        line = line.strip().split(' ')
        src[line[0]] = line[1]

import shutil
total = 0
with open(tgt_scp) as f_t:
    for line in f_t:
        line = line.strip().split(' ')
        name = line[0]
        path = line[1]
        if name in src and path != src[name]:
            shutil.copy(src[name], path)
            total+=1
print(f'{total} files are copied.')

