#!/usr/bin/env bash
experiment=$1

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#set=S    # S for the small set, M for the mediate set, L for the large set
train_set=train
valid_set=dev
test_sets="test_aishell test_libriclean test_meeting test_net test_giga test_talcs"



if [ $experiment -eq 0 ]; then
stage=7
stop_stage=8
expdir=exp/hubert-base
#asr_config=conf/train_asr_conformer_hubert-base.yaml
asr_config=conf/train_asr_hubert_base.yaml
inference_config=conf/decode_asr_transformer_ngram.yaml
ngram_num=4
lm_config=conf/train_lm.yaml
use_lm=true
use_ngram=true
# speed perturbation related
# add "--speed_perturb_factors="0.9 1.0 1.1" if you want to
# apply speed perturbation for the training data

CUDA_VISIBLE_DEVICES=2,3                           \
./asr.sh                                               \
    --ngpu 2                                           \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --expdir ${expdir}                                 \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type bpe                                   \
    --bpemode  bpe                                     \
    --nbpe 16000                                       \
    --num_splits_lm  8                                 \
    --num_splits_asr 8                                 \
    --feats_normalize null                             \
    --use_lm ${use_lm}                                 \
    --lm_config "${lm_config}"                         \
    --use_ngram "${use_ngram}"                         \
    --ngram_num   ${ngram_num}                         \
    --inference_ngram ${ngram_num}gram.bin             \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_nj 16                                   \
    --gpu_inference true                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/train/text" #"$@"

fi

if [ $experiment -eq 1 ]; then
stage=11
stop_stage=13
expdir=exp/hubert-base
#asr_config=conf/train_asr_conformer_hubert-base.yaml
asr_config=conf/train_asr_hubert_base.yaml
inference_config=conf/decode_asr_transformer_ngram.yaml
ngram_num=4
lm_config=conf/train_lm.yaml
use_lm=true
use_ngram=true
# speed perturbation related
# add "--speed_perturb_factors="0.9 1.0 1.1" if you want to
# apply speed perturbation for the training data

CUDA_VISIBLE_DEVICES=4,5,6,7                           \
./asr.sh                                               \
    --ngpu 4                                           \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --expdir ${expdir}                                 \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type bpe                                   \
    --bpemode  bpe                                     \
    --nbpe 16000                                       \
    --num_splits_lm  8                                 \
    --num_splits_asr 8                                 \
    --feats_normalize null                             \
    --use_lm ${use_lm}                                 \
    --lm_config "${lm_config}"                         \
    --use_ngram "${use_ngram}"                         \
    --ngram_num   ${ngram_num}                         \
    --inference_ngram ${ngram_num}gram.bin             \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_nj 16                                   \
    --gpu_inference true                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/train/text" 

fi

