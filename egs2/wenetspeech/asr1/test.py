# - * - coding:utf-8 - * -
import sys
import os

sys.path.append("/data/megastore/Projects/DuJing/code/espnet/")
from espnet2.bin import asr_train, asr_inference
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(os.getcwd())
os.chdir("/data/megastore/Projects/DuJing/code/espnet/egs2/wenetspeech/asr1")
print(os.getcwd())
# 训练代码
param="--use_preprocessor true --src_bpemodel data/zhen_tc_punc_char_token_list/src_bpe_char8000/bpe.model --src_token_type bpe --src_token_list data/zhen_tc_punc_char_token_list/src_bpe_char8000/tokens.txt --bpemodel data/zhen_tc_punc_char_token_list/tgt_bpe_bpe8000/bpe.model --token_type bpe --token_list data/zhen_tc_punc_char_token_list/tgt_bpe_bpe8000/tokens.txt --non_linguistic_symbols none --cleaner none --g2p none --valid_data_path_and_name_and_type dump/fbank/dev/feats.scp,speech,kaldi_ark --valid_data_path_and_name_and_type dump/fbank/dev/text.en,text,text --valid_data_path_and_name_and_type dump/fbank/dev/text.zh,src_text,text --valid_shape_file exp/st_stats_fbank_zhen_tc_punc_char_bpe8000/valid/speech_shape --valid_shape_file exp/st_stats_fbank_zhen_tc_punc_char_bpe8000/valid/text_shape --valid_shape_file exp/st_stats_fbank_zhen_tc_punc_char_bpe8000/valid/src_text_shape --resume true --fold_length 800 --fold_length 150 --output_dir exp/st_train_st_conformer_mtl2_fbank_zhen_tc_punc_char_bpe8000 --config conf/train_st_conformer_mtl2.yaml --input_size=80 --normalize=global_mvn --normalize_conf stats_file=exp/st_stats_fbank_zhen_tc_punc_char_bpe8000/train/feats_stats.npz --train_data_path_and_name_and_type dump/fbank/train/feats.scp,speech,kaldi_ark --train_data_path_and_name_and_type dump/fbank/train/text.en,text,text --train_data_path_and_name_and_type dump/fbank/train/text.zh,src_text,text --train_shape_file exp/st_stats_fbank_zhen_tc_punc_char_bpe8000/train/speech_shape --train_shape_file exp/st_stats_fbank_zhen_tc_punc_char_bpe8000/train/text_shape --train_shape_file exp/st_stats_fbank_zhen_tc_punc_char_bpe8000/train/src_text_shape --ngpu 1 --multiprocessing_distributed True "

#asr_train.main(param)

#param1 = "--batch_size 1 --ngpu 0 --data_path_and_name_and_type dump/raw/ada/wav.scp,speech,sound --key_file exp/asr_train_asr_raw_zh_char/decode_asr_asr_model_valid.acc.ave_10best/ada/logdir/keys.1.scp --asr_train_config exp/asr_train_asr_raw_zh_char/config.yaml --asr_model_file exp/asr_train_asr_raw_zh_char/valid.acc.ave_10best.pth --output_dir exp/asr_train_asr_raw_zh_char/decode_asr_asr_model_valid.acc.ave_10best/ada/logdir/output.1 --config conf/decode_asr.yaml"
asr_inference.main()
