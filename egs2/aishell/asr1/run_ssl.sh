#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#set=S    # S for the small set, M for the mediate set, L for the large set
train_set=train
valid_set=dev
test_sets="dev test"

experiment=$1

# 使用hubert_encoder + transformer_decoder
if [ $experiment -eq 0 ]; then
stage=12
stop_stage=13
expdir=exp/hubert-base
asr_config=conf/train_asr_hubert_base.yaml
inference_config=conf/decode_asr_transformer_ngram.yaml
lm_config=conf/train_lm.yaml
use_lm=false   #true
use_ngram=true
inference_asr_model=valid.acc.ave_10best.pth
ngram_num=4
# speed perturbation related
# add "--speed_perturb_factors="0.9 1.0 1.1" if you want to
# apply speed perturbation for the training data

CUDA_VISIBLE_DEVICES=7                                 \
./asr.sh                                               \
    --ngpu 1                                           \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --expdir ${expdir}                                 \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --feats_normalize null                             \
    --use_lm ${use_lm}                                 \
    --lm_config "${lm_config}"                         \
    --use_ngram "${use_ngram}"                         \
    --ngram_num   ${ngram_num}                         \
    --inference_ngram ${ngram_num}gram.bin             \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_nj 16                                  \
    --inference_asr_model ${inference_asr_model}       \
    --gpu_inference false                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/train/text" #"$@"

fi

# 使用hubert_encoder + transformer_decoder, batch不按照音频长度排序
if [ $experiment -eq 1 ]; then
stage=12
stop_stage=13
expdir=exp/hubert-base
asr_config=conf/train_asr_hubert_base_nonsort.yaml
inference_config=conf/decode_asr_transformer_ngram.yaml
lm_config=conf/train_lm.yaml
use_lm=true
use_ngram=true
inference_asr_model=valid.acc.ave_10best.pth
ngram_num=4
# speed perturbation related
# add "--speed_perturb_factors="0.9 1.0 1.1" if you want to
# apply speed perturbation for the training data

CUDA_VISIBLE_DEVICES=7                                 \
./asr.sh                                               \
    --ngpu 1                                           \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --expdir ${expdir}                                 \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --feats_normalize null                             \
    --use_lm ${use_lm}                                 \
    --lm_config "${lm_config}"                         \
    --use_ngram "${use_ngram}"                         \
    --ngram_num   ${ngram_num}                         \
    --inference_ngram ${ngram_num}gram.bin             \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_nj 16                                  \
    --inference_asr_model ${inference_asr_model}       \
    --gpu_inference false                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/train/text" 

fi

# 使用branchformer_encoder + transformer_decoder
if [ $experiment -eq 2 ]; then
stage=12
stop_stage=13
expdir=exp/hubert-base
asr_config=conf/train_asr_branchformer.yaml
inference_config=conf/decode_asr_transformer_ngram.yaml
lm_config=conf/train_lm.yaml
use_lm=false   #true
use_ngram=true
inference_asr_model=valid.acc.ave_10best.pth
ngram_num=4
# speed perturbation related
# add "--speed_perturb_factors="0.9 1.0 1.1" if you want to
# apply speed perturbation for the training data

CUDA_VISIBLE_DEVICES=6                                 \
./asr.sh                                               \
    --ngpu 1                                           \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --expdir ${expdir}                                 \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --feats_normalize null                             \
    --use_lm ${use_lm}                                 \
    --lm_config "${lm_config}"                         \
    --use_ngram "${use_ngram}"                         \
    --ngram_num   ${ngram_num}                         \
    --inference_ngram ${ngram_num}gram.bin             \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_nj 16                                   \
    --inference_asr_model ${inference_asr_model}       \
    --gpu_inference false                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/train/text" #"$@"

fi

# hubert作为特征提取器，使用branchformer_encoder + transformer_decoder
if [ $experiment -eq 3 ]; then
stage=12
stop_stage=13
expdir=exp/hubert-base
asr_config=conf/train_asr_hubert_branchformer.yaml
inference_config=conf/decode_asr_transformer_ngram.yaml
lm_config=conf/train_lm.yaml
use_lm=false   #true
use_ngram=true
inference_asr_model=valid.acc.ave_10best.pth
ngram_num=4
# speed perturbation related
# add "--speed_perturb_factors="0.9 1.0 1.1" if you want to
# apply speed perturbation for the training data

CUDA_VISIBLE_DEVICES=2,3                                 \
./asr.sh                                               \
    --ngpu 2                                           \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --expdir ${expdir}                                 \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --feats_normalize null                             \
    --use_lm ${use_lm}                                 \
    --lm_config "${lm_config}"                         \
    --use_ngram "${use_ngram}"                         \
    --ngram_num   ${ngram_num}                         \
    --inference_ngram ${ngram_num}gram.bin             \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_nj 16                                   \
    --inference_asr_model ${inference_asr_model}       \
    --gpu_inference false                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/train/text" #"$@"

fi
# 更新hubert的conformer层，使用branchformer_encoder + transformer_decoder
if [ $experiment -eq 4 ]; then
stage=12
stop_stage=13
expdir=exp/hubert-base
asr_config=conf/train_asr_hubert_ft_branchformer.yaml
inference_config=conf/decode_asr_transformer_ngram.yaml
lm_config=conf/train_lm.yaml
use_lm=false   #true
use_ngram=false
inference_asr_model=valid.acc.ave_10best.pth
ngram_num=4
# speed perturbation related
# add "--speed_perturb_factors="0.9 1.0 1.1" if you want to
# apply speed perturbation for the training data

CUDA_VISIBLE_DEVICES=3                                 \
./asr.sh                                               \
    --ngpu 1                                           \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --expdir ${expdir}                                 \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --feats_normalize null                             \
    --use_lm ${use_lm}                                 \
    --lm_config "${lm_config}"                         \
    --use_ngram "${use_ngram}"                         \
    --ngram_num   ${ngram_num}                         \
    --inference_ngram ${ngram_num}gram.bin             \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_nj 32                                   \
    --inference_asr_model ${inference_asr_model}       \
    --gpu_inference false                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/train/text" #"$@"

fi
