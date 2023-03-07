#!/usr/bin/env bash
experiment=$1

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set=train
valid_set=dev
test_sets="test_aishell test_net test_meeting test_libriclean  test_giga
test_talcs test_htrs462 test_sjtcs test_conv test_xmov test_xmov_inter"

# 训练语言模型
if [ $experiment -eq 0 ]; then
stage=7
stop_stage=8
expdir=exp/hubert-base
#asr_config=conf/train_asr_conformer_hubert-base.yaml
asr_config=conf/train_asr_wavaug_hubert_specaug_subsamp.yaml
inference_config=conf/decode_asr_transformer_ngram.yaml
inference_asr_model=valid.acc.ave_10best.pth
lm_config=conf/train_lm.yaml
inference_lm=valid.loss.ave_10best.pth
use_lm=true
ngram_num=3
use_ngram=false
# speed perturbation related
# add "--speed_perturb_factors="0.9 1.0 1.1" if you want to
# apply speed perturbation for the training data

if [ ! -f "data/train/bpe_nlsyms.txt" ]; then
    # must preprocess data first to get Mandarin character tokens
    if [ ${stage} -eq 1 ]; then
        ./asr.sh \
            --train_set "${train_set}" \
            --valid_set "${valid_set}" \
            --test_sets "${test_sets}" \
            --stage 1 --stop_stage 1
        stage=2
    else
        echo "Error: data/train/bpe_nlsyms.txt does not exist! Run from stage=1 again."
        exit 1
    fi
fi

bpe_nlsyms=data/train/bpe_nlsyms.txt

CUDA_VISIBLE_DEVICES=4,5,6,7                           \
./asr.sh                                               \
    --ngpu 4                                           \
    --nj   48                                          \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --expdir ${expdir}                                 \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type bpe                                   \
    --bpemode  bpe                                     \
    --bpe_train_text "data/${train_set}/bpe_train.txt"          \
    --bpe_nlsyms "${bpe_nlsyms}"                       \
    --nbpe 12000                                       \
    --nlsyms_txt data/non_linguistic_symbols.txt       \
    --num_splits_lm  128                                 \
    --num_splits_asr 8                                 \
    --feats_normalize null                             \
    --use_lm ${use_lm}                                 \
    --lm_config "${lm_config}"                         \
    --inference_lm ${inference_lm}                     \
    --use_ngram "${use_ngram}"                         \
    --ngram_num   ${ngram_num}                         \
    --inference_ngram ${ngram_num}gram.bin             \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_asr_model ${inference_asr_model}       \
    --inference_nj 16                                   \
    --gpu_inference false                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/lm_corpus/text"


fi

# 使用中文字+英文bpe=12000词典
if [ $experiment -eq 1 ]; then
stage=12
stop_stage=13
expdir=exp/hubert-base
#asr_config=conf/train_asr_conformer_hubert-base.yaml
asr_config=conf/train_asr_wavaug_hubert_specaug_subsamp.yaml
inference_config=conf/decode_asr_transformer_ngram.yaml
inference_asr_model=valid.acc.ave_10best.pth
lm_config=conf/train_lm.yaml
inference_lm=valid.loss.ave_10best.pth
use_lm=false
ngram_num=3
use_ngram=false
# speed perturbation related
# add "--speed_perturb_factors="0.9 1.0 1.1" if you want to
# apply speed perturbation for the training data

if [ ! -f "data/train/bpe_nlsyms.txt" ]; then
    # must preprocess data first to get Mandarin character tokens
    if [ ${stage} -eq 1 ]; then
        ./asr.sh \
            --train_set "${train_set}" \
            --valid_set "${valid_set}" \
            --test_sets "${test_sets}" \
            --stage 1 --stop_stage 1
        stage=2
    else
        echo "Error: data/train/bpe_nlsyms.txt does not exist! Run from stage=1 again."
        exit 1
    fi
fi

bpe_nlsyms=data/train/bpe_nlsyms.txt

CUDA_VISIBLE_DEVICES=0,1,2,3                                 \
./asr.sh                                               \
    --ngpu 4                                           \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --expdir ${expdir}                                 \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type bpe                                   \
    --bpemode  bpe                                     \
    --bpe_train_text "data/${train_set}/bpe_train.txt"          \
    --bpe_nlsyms "${bpe_nlsyms}"                       \
    --nbpe 12000                                       \
    --nlsyms_txt data/non_linguistic_symbols.txt       \
    --num_splits_lm  128                                 \
    --num_splits_asr 8                                 \
    --feats_normalize null                             \
    --use_lm ${use_lm}                                 \
    --lm_config "${lm_config}"                         \
    --inference_lm ${inference_lm}                     \
    --use_ngram "${use_ngram}"                         \
    --ngram_num   ${ngram_num}                         \
    --inference_ngram ${ngram_num}gram.bin             \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_asr_model ${inference_asr_model}       \
    --inference_nj 16                                   \
    --gpu_inference false                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/lm_corpus/text" 

fi

# 使用中文字+英文bpe=12000词典，帧率为40
if [ $experiment -eq 2 ]; then
test_sets="test_fleurs "
stage=12
stop_stage=13
expdir=exp/hubert-base
#asr_config=conf/train_asr_conformer_hubert-base.yaml
asr_config=conf/train_asr_wavaug_hubert_specaug_2subsamp.yaml
inference_config=conf/decode_asr_transformer_ngram.yaml
inference_asr_model=valid.acc.ave_10best.pth
lm_config=conf/train_lm.yaml
inference_lm=valid.loss.ave_10best.pth
use_lm=false
ngram_num=3
use_ngram=false
# speed perturbation related
# add "--speed_perturb_factors="0.9 1.0 1.1" if you want to
# apply speed perturbation for the training data

if [ ! -f "data/train/bpe_nlsyms.txt" ]; then
    # must preprocess data first to get Mandarin character tokens
    if [ ${stage} -eq 1 ]; then
        ./asr.sh \
            --train_set "${train_set}" \
            --valid_set "${valid_set}" \
            --test_sets "${test_sets}" \
            --stage 1 --stop_stage 1
        stage=2
    else
        echo "Error: data/train/bpe_nlsyms.txt does not exist! Run from stage=1 again."
        exit 1
    fi
fi

bpe_nlsyms=data/train/bpe_nlsyms.txt

CUDA_VISIBLE_DEVICES=0,1,2,3                                 \
./asr.sh                                               \
    --ngpu 4                                           \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --expdir ${expdir}                                 \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type bpe                                   \
    --bpemode  bpe                                     \
    --bpe_train_text "data/${train_set}/bpe_train.txt"          \
    --bpe_nlsyms "${bpe_nlsyms}"                       \
    --nbpe 12000                                       \
    --nlsyms_txt data/non_linguistic_symbols.txt       \
    --num_splits_lm  128                                 \
    --num_splits_asr 8                                 \
    --feats_normalize null                             \
    --use_lm ${use_lm}                                 \
    --lm_config "${lm_config}"                         \
    --inference_lm ${inference_lm}                     \
    --use_ngram "${use_ngram}"                         \
    --ngram_num   ${ngram_num}                         \
    --inference_ngram ${ngram_num}gram.bin             \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_asr_model ${inference_asr_model}       \
    --inference_nj 32                                   \
    --gpu_inference false                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/lm_corpus/text" 

fi

# 使用中文字+英文bpe=12000词典，帧率为40，解码时加入热词
if [ $experiment -eq 3 ]; then
test_sets="test_xmov "
stage=12
stop_stage=13
expdir=exp/hubert-base
#asr_config=conf/train_asr_conformer_hubert-base.yaml
asr_config=conf/train_asr_wavaug_hubert_specaug_2subsamp.yaml
inference_config=conf/decode_asr_transformer_ngram_hotword.yaml
inference_asr_model=valid.acc.ave_10best.pth
lm_config=conf/train_lm.yaml
inference_lm=valid.loss.ave_10best.pth
use_lm=false
ngram_num=3
use_ngram=false
# speed perturbation related
# add "--speed_perturb_factors="0.9 1.0 1.1" if you want to
# apply speed perturbation for the training data

if [ ! -f "data/train/bpe_nlsyms.txt" ]; then
    # must preprocess data first to get Mandarin character tokens
    if [ ${stage} -eq 1 ]; then
        ./asr.sh \
            --train_set "${train_set}" \
            --valid_set "${valid_set}" \
            --test_sets "${test_sets}" \
            --stage 1 --stop_stage 1
        stage=2
    else
        echo "Error: data/train/bpe_nlsyms.txt does not exist! Run from stage=1 again."
        exit 1
    fi
fi

bpe_nlsyms=data/train/bpe_nlsyms.txt

CUDA_VISIBLE_DEVICES=0,1,2,3                                 \
./asr.sh                                               \
    --ngpu 4                                           \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --expdir ${expdir}                                 \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type bpe                                   \
    --bpemode  bpe                                     \
    --bpe_train_text "data/${train_set}/bpe_train.txt"          \
    --bpe_nlsyms "${bpe_nlsyms}"                       \
    --nbpe 12000                                       \
    --nlsyms_txt data/non_linguistic_symbols.txt       \
    --num_splits_lm  128                                 \
    --num_splits_asr 8                                 \
    --feats_normalize null                             \
    --use_lm ${use_lm}                                 \
    --lm_config "${lm_config}"                         \
    --inference_lm ${inference_lm}                     \
    --use_ngram "${use_ngram}"                         \
    --ngram_num   ${ngram_num}                         \
    --inference_ngram ${ngram_num}gram.bin             \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_asr_model ${inference_asr_model}       \
    --inference_nj 16                                   \
    --gpu_inference false                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/lm_corpus/text" 

fi


# 使用中文字+英文bpe=12000词典， 帧率为40，流式模型
if [ $experiment -eq 4 ]; then
#test_sets="test "
stage=12
stop_stage=13
expdir=exp/hubert-base
#asr_config=conf/train_asr_conformer_hubert-base.yaml
asr_config=conf/train_asr_streaming_conformer_40fps_4lf.yaml
inference_config=conf/decode_asr_streaming.yaml
inference_asr_model=valid.acc.ave_10best.pth
lm_config=conf/train_lm.yaml
inference_lm=valid.loss.ave_10best.pth
use_lm=false
ngram_num=3
use_ngram=false
# speed perturbation related
# add "--speed_perturb_factors="0.9 1.0 1.1" if you want to
# apply speed perturbation for the training data

if [ ! -f "data/train/bpe_nlsyms.txt" ]; then
    # must preprocess data first to get Mandarin character tokens
    if [ ${stage} -eq 1 ]; then
        ./asr.sh \
            --train_set "${train_set}" \
            --valid_set "${valid_set}" \
            --test_sets "${test_sets}" \
            --stage 1 --stop_stage 1
        stage=2
    else
        echo "Error: data/train/bpe_nlsyms.txt does not exist! Run from stage=1 again."
        exit 1
    fi
fi

bpe_nlsyms=data/train/bpe_nlsyms.txt

CUDA_VISIBLE_DEVICES=0,1,2,3                                 \
./asr.sh                                               \
    --ngpu 4                                           \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --expdir ${expdir}                                 \
    --use_streaming true                               \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type bpe                                   \
    --bpemode  bpe                                     \
    --bpe_train_text "data/${train_set}/bpe_train.txt"          \
    --bpe_nlsyms "${bpe_nlsyms}"                       \
    --nbpe 12000                                       \
    --nlsyms_txt data/non_linguistic_symbols.txt       \
    --num_splits_lm  128                                 \
    --num_splits_asr 8                                 \
    --feats_normalize none                             \
    --use_lm ${use_lm}                                 \
    --lm_config "${lm_config}"                         \
    --inference_lm ${inference_lm}                     \
    --use_ngram "${use_ngram}"                         \
    --ngram_num   ${ngram_num}                         \
    --inference_ngram ${ngram_num}gram.bin             \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_asr_model ${inference_asr_model}       \
    --inference_nj 32                                   \
    --gpu_inference false                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/lm_corpus/text" 

fi

# 使用中文字+英文bpe=12000词典， 帧率为40，流式模型
if [ $experiment -eq 5 ]; then
#test_sets="test_aishell "
stage=12
stop_stage=13
expdir=exp/hubert-base
#asr_config=conf/train_asr_conformer_hubert-base.yaml
asr_config=conf/train_asr_streaming_conformer2_40fps_8lf.yaml
inference_config=conf/decode_asr_streaming.yaml
inference_asr_model=valid.acc.ave_10best.pth
#inference_asr_model=valid.acc.best.pth
lm_config=conf/train_lm.yaml
inference_lm=valid.loss.ave_10best.pth
use_lm=false
ngram_num=3
use_ngram=false
# speed perturbation related
# add "--speed_perturb_factors="0.9 1.0 1.1" if you want to
# apply speed perturbation for the training data

if [ ! -f "data/train/bpe_nlsyms.txt" ]; then
    # must preprocess data first to get Mandarin character tokens
    if [ ${stage} -eq 1 ]; then
        ./asr.sh \
            --train_set "${train_set}" \
            --valid_set "${valid_set}" \
            --test_sets "${test_sets}" \
            --stage 1 --stop_stage 1
        stage=2
    else
        echo "Error: data/train/bpe_nlsyms.txt does not exist! Run from stage=1 again."
        exit 1
    fi
fi

bpe_nlsyms=data/train/bpe_nlsyms.txt

CUDA_VISIBLE_DEVICES=0,1,2,3                                 \
./asr.sh                                               \
    --ngpu 4                                           \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --expdir ${expdir}                                 \
    --use_streaming true                               \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type bpe                                   \
    --bpemode  bpe                                     \
    --bpe_train_text "data/${train_set}/bpe_train.txt"          \
    --bpe_nlsyms "${bpe_nlsyms}"                       \
    --nbpe 12000                                       \
    --nlsyms_txt data/non_linguistic_symbols.txt       \
    --num_splits_lm  128                                 \
    --num_splits_asr 8                                 \
    --feats_normalize none                             \
    --use_lm ${use_lm}                                 \
    --lm_config "${lm_config}"                         \
    --inference_lm ${inference_lm}                     \
    --use_ngram "${use_ngram}"                         \
    --ngram_num   ${ngram_num}                         \
    --inference_ngram ${ngram_num}gram.bin             \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_asr_model ${inference_asr_model}       \
    --inference_nj 16                                   \
    --gpu_inference false                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/lm_corpus/text" 

fi

