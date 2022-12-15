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
asr_config=conf/train_asr_hubert_base.yaml
inference_config=conf/decode_asr_transformer_ngram.yaml
inference_asr_model=valid.acc.ave_10best.pth
lm_config=conf/train_lm.yaml
use_lm=false
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

CUDA_VISIBLE_DEVICES=0                                 \
./asr.sh                                               \
    --stage 12                                         \
    --stop_stage 13                                    \
    --use_streaming false                               \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --feats_normalize null                             \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_nj 16                                   \
    --inference_asr_model ${inference_asr_model}       \
    --gpu_inference false                               \
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
asr_config=conf/train_asr_wavaug_hubert_base.yaml
inference_config=conf/decode_asr_transformer_ngram.yaml
inference_asr_model=valid.acc.ave_10best.pth
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
    --use_streaming false                               \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --feats_normalize null                             \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_nj 16                                   \
    --inference_asr_model ${inference_asr_model}       \
    --gpu_inference false                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" #"$@"
    
fi

if [ $experiment -eq 2 ]; then
asr_config=conf/train_asr_hubert_specaug_base.yaml
inference_config=conf/decode_asr_transformer_ngram.yaml
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
    --use_streaming false                               \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --feats_normalize null                             \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_nj 16                                   \
    --inference_asr_model ${inference_asr_model}       \
    --gpu_inference false                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" #"$@"
    
fi

if [ $experiment -eq 3 ]; then
asr_config=conf/train_asr_hubert_downsample_base.yaml
inference_config=conf/decode_asr_transformer_ngram.yaml
inference_asr_model=valid.acc.ave_10best.pth
lm_config=conf/train_lm.yaml
use_lm=false
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

CUDA_VISIBLE_DEVICES=3                                 \
./asr.sh                                               \
    --stage 12                                         \
    --stop_stage 13                                    \
    --use_streaming false                               \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --feats_normalize null                             \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_nj 16                                   \
    --inference_asr_model ${inference_asr_model}       \
    --gpu_inference false                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" #"$@"
    
fi

if [ $experiment -eq 4 ]; then
asr_config=conf/train_asr_wavaug_hubert_specaug_subsamp.yaml
inference_config=conf/decode_asr_transformer_ngram.yaml
inference_asr_model=valid.acc.ave_10best.pth
lm_config=conf/train_lm.yaml
use_lm=false
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

CUDA_VISIBLE_DEVICES=5                                 \
./asr.sh                                               \
    --stage 12                                         \
    --stop_stage 13                                    \
    --use_streaming false                               \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --feats_normalize null                             \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_nj 16                                   \
    --inference_asr_model ${inference_asr_model}       \
    --gpu_inference false                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" #"$@"
    
fi
