#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# environment: fairseq, run this in command-window first
# conda activate fairseq

espnet_dir=/data/megastore/Projects/DuJing/code/espnet
egs_path=$espnet_dir/egs2/xmov_cs/asr2
asr_config=$egs_path/exp/hubert-base/asr_train_asr_wavaug_hubert_specaug_subsamp_raw_bpe12000/config.yaml
asr_model=$egs_path/exp/hubert-base/asr_train_asr_wavaug_hubert_specaug_subsamp_raw_bpe12000/valid.acc.ave_10best.pth
bpe_model=$egs_path/data/token_list/bpe_bpe12000/bpe.model

wav_path=/data/megastore/Datasets/ASR/ManEngMix/DataTang_CS/data/category/G86020/G86020S1001.wav
text_path=./text           # text with multiple lines where each unit is align basis: Chinese char and English word
result_path=./output.txt

CUDA_VISIBLE_DEVICES=                           \
$espnet_dir/espnet2/bin/asr_align.py   \
    --ngpu 0                                                            \
    --kaldi_style_text False                                            \
    --token_type  bpe                                                   \
    --bpemodel   $bpe_model                                             \
    --asr_train_config "${asr_config}"                                  \
    --asr_model_file ${asr_model}                                       \
    --audio     $wav_path                                               \
    --text    $text_path                                               \
    -o        $result_path

