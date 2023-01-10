stage=13
stop_stage=13
token_type=bpe
test_sets="test_xmov1 "

python=python3
data_feats=dump/raw
asr_exp=exp/hubert-base/asr_train_asr_wavaug_hubert_specaug_2subsamp_raw_bpe12000
inference_tag=decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best
cleaner=none
nlsyms_txt=none
bpemodel=data/token_list/bpe_bpe12000/bpe.model

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    echo "Stage 13: Scoring"
    if [ "${token_type}" = phn ]; then
        echo "Error: Not implemented for token_type=phn"
        exit 1
    fi

    for dset in ${test_sets}; do
        _data="${data_feats}/${dset}"
        _dir="${asr_exp}/${inference_tag}/${dset}"

        for _type in cer wer ter; do
            [ "${_type}" = ter ] && [ ! -f "${bpemodel}" ] && continue

            _scoredir="${_dir}/score_${_type}"
            mkdir -p "${_scoredir}"

            if [ "${_type}" = wer ]; then
                # Tokenize text to word level
                paste \
                    <(<"${_data}/text" \
                          ${python} -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type word \
                              --non_linguistic_symbols "${nlsyms_txt}" \
                              --remove_non_linguistic_symbols true \
                              --cleaner "${cleaner}" \
                              ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/ref.trn"

                # NOTE(kamo): Don't use cleaner for hyp
                paste \
                    <(<"${_dir}/text"  \
                          ${python} -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type word \
                              --non_linguistic_symbols "${nlsyms_txt}" \
                              --remove_non_linguistic_symbols true \
                              ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/hyp.trn"


            elif [ "${_type}" = cer ]; then
                # Tokenize text to char level
                paste \
                    <(<"${_data}/text" \
                          ${python} -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type char \
                              --non_linguistic_symbols "${nlsyms_txt}" \
                              --remove_non_linguistic_symbols true \
                              --cleaner "${cleaner}" \
                              ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/ref.trn"

                # NOTE(kamo): Don't use cleaner for hyp
                paste \
                    <(<"${_dir}/text"  \
                          ${python} -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type char \
                              --non_linguistic_symbols "${nlsyms_txt}" \
                              --remove_non_linguistic_symbols true \
                              ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/hyp.trn"

            elif [ "${_type}" = ter ]; then
                # Tokenize text using BPE
                paste \
                    <(<"${_data}/text" \
                          ${python} -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type bpe \
                              --bpemodel "${bpemodel}" \
                              --cleaner "${cleaner}" \
                            ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/ref.trn"

                # NOTE(kamo): Don't use cleaner for hyp
                paste \
                    <(<"${_dir}/text" \
                          ${python} -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type bpe \
                              --bpemodel "${bpemodel}" \
                              ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/hyp.trn"

            fi

            sclite \
    ${score_opts} \
                -r "${_scoredir}/ref.trn" trn \
                -h "${_scoredir}/hyp.trn" trn \
                -i rm -o all stdout > "${_scoredir}/result.txt"

            echo "Write ${_type} result in ${_scoredir}/result.txt"
            grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
        done
    done
fi