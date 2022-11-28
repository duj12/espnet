#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=1
stop_stage=100000

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Preparation"
    # for bpe training
    echo "Preparing bpe train text."
    mv data/train/text data/train/text0
    local/preprocess.py data/train/text0 data/train/text data/train/chars.txt data/train/bpe_nlsyms.txt data/train/bpe_train.txt
    
    #insert space in Mandarin text data
    for set in dev test_aishell test_libriclean test_meeting test_net test_giga test_talcs test_htrs462 test_sjtcs ; do 
        echo "add space in Chinese chars of "${set}
        mv data/$set/text data/$set/text0
        local/add_space_between_chinese.py data/$set/text0  data/$set/text
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
