#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

experiment=$1

train_set=train
valid_set=dev
test_sets="dev test"


if [ $experiment -eq 0 ]; then
asr_config=conf/train_asr_streaming_conformer.yaml
inference_config=conf/decode_asr_streaming.yaml
inference_asr_model=valid.acc.ave_10best.pth
lm_config=conf/train_lm.yaml
use_lm=false
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

CUDA_VISIBLE_DEVICES=2                                 \
./asr.sh                                               \
    --stage 12                                         \
    --stop_stage 13                                    \
    --use_streaming true                               \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_nj 16                                   \
    --inference_asr_model ${inference_asr_model}       \
    --gpu_inference true                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" #"$@"
    
fi


if [ $experiment -eq 1 ]; then
asr_config=conf/train_asr_streaming_conformer_rnn_transducer.yaml
inference_config=conf/decode_asr_rnnt.yaml
inference_asr_model=valid.loss.ave_10best.pth
#inference_asr_model=valid.loss.ave.pth
lm_config=conf/train_lm.yaml
use_lm=false
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

CUDA_VISIBLE_DEVICES=1                                 \
./asr.sh                                               \
    --stage 12                                         \
    --stop_stage 13                                    \
    --use_streaming true                               \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_nj 16                                   \
    --inference_asr_model ${inference_asr_model}       \
    --gpu_inference true                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" #"$@"
    
fi