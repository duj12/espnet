# Hubert encoder RESULTS
## Environments
- date: `Tue Nov 22 21:36:33 CST 2022`
- python version: `3.7.0 | packaged by conda-forge | (default, Nov 12 2018, 20:15:55)  [GCC 7.3.0]`
- espnet version: `espnet 202205`
- pytorch version: `pytorch 1.11.0`
- Git hash: ``
  - Commit date: ``

## asr_train_asr_hubert_base_raw_zh_char_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/dev|14326|14326|67.2|32.8|0.0|0.0|32.8|32.8|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/test|7176|7176|64.3|35.7|0.0|0.0|35.7|35.7|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/dev|14326|205341|96.1|3.9|0.1|0.1|4.0|32.8|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/test|7176|104765|95.6|4.3|0.1|0.1|4.5|35.7|

# Hubert encoder + WaveAug RESULTS
## Environments
- date: `Wed Nov 23 12:24:00 CST 2022`
- python version: `3.7.0 | packaged by conda-forge | (default, Nov 12 2018, 20:15:55)  [GCC 7.3.0]`
- espnet version: `espnet 202205`
- pytorch version: `pytorch 1.11.0`
- Git hash: ``
  - Commit date: ``

## asr_train_asr_wavaug_hubert_base_raw_zh_char_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/dev|14326|14326|66.3|33.7|0.0|0.0|33.7|33.7|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/test|7176|7176|65.0|35.0|0.0|0.0|35.0|35.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/dev|14326|205341|96.0|4.0|0.1|0.1|4.1|33.7|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/test|7176|104765|95.7|4.2|0.1|0.1|4.4|35.0|

# Hubert encoder + SpecAug RESULTS
## Environments
- date: `Wed Nov 23 12:31:09 CST 2022`
- python version: `3.7.0 | packaged by conda-forge | (default, Nov 12 2018, 20:15:55)  [GCC 7.3.0]`
- espnet version: `espnet 202205`
- pytorch version: `pytorch 1.11.0`
- Git hash: ``
  - Commit date: ``

## asr_train_asr_hubert_specaug_base_raw_zh_char_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/dev|14326|14326|67.1|32.9|0.0|0.0|32.9|32.9|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/test|7176|7176|64.9|35.1|0.0|0.0|35.1|35.1|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/dev|14326|205341|96.1|3.8|0.1|0.1|4.0|32.9|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/test|7176|104765|95.7|4.2|0.1|0.1|4.4|35.1|

# Hubert encoder + SubSampling RESULTS
## Environments
- date: `Wed Nov 23 12:46:20 CST 2022`
- python version: `3.7.0 | packaged by conda-forge | (default, Nov 12 2018, 20:15:55)  [GCC 7.3.0]`
- espnet version: `espnet 202205`
- pytorch version: `pytorch 1.11.0`
- Git hash: ``
  - Commit date: ``

## asr_train_asr_hubert_downsample_base_raw_zh_char_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/dev|14326|14326|69.4|30.6|0.0|0.0|30.6|30.6|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/test|7176|7176|67.2|32.8|0.0|0.0|32.8|32.8|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/dev|14326|205341|96.4|3.5|0.1|0.1|3.7|30.6|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/test|7176|104765|96.1|3.8|0.1|0.1|4.0|32.8|

# Streaming Conformer RESULTS
## Environments
- date: `Wed Nov 16 00:00:03 CST 2022`
- python version: `3.7.0 | packaged by conda-forge | (default, Nov 12 2018, 20:15:55)  [GCC 7.3.0]`
- espnet version: `espnet 202205`
- pytorch version: `pytorch 1.11.0`
- Git hash: ``
  - Commit date: ``

## asr_train_asr_streaming_conformer_raw_zh_char_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_streaming_asr_model_valid.acc.ave_10best/dev|14326|14326|59.7|40.3|0.0|0.0|40.3|40.3|
|decode_asr_streaming_asr_model_valid.acc.ave_10best/test|7176|7176|57.0|43.0|0.0|0.0|43.0|43.0|
|decode_asr_streaming_lm_lm_train_lm_zh_char_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|14326|62.4|37.6|0.0|0.0|37.6|37.6|
|decode_asr_streaming_lm_lm_train_lm_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test|7176|7176|60.4|39.6|0.0|0.0|39.6|39.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_streaming_asr_model_valid.acc.ave_10best/dev|14326|205341|94.8|5.0|0.1|0.3|5.4|40.3|
|decode_asr_streaming_asr_model_valid.acc.ave_10best/test|7176|104765|94.1|5.6|0.3|0.2|6.1|43.0|
|decode_asr_streaming_lm_lm_train_lm_zh_char_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|95.0|4.8|0.1|0.2|5.1|37.6|
|decode_asr_streaming_lm_lm_train_lm_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|94.5|5.2|0.3|0.2|5.7|39.6|

# Streaming Conformer + RNNT RESULTS
## Environments
- date: `Wed Nov 16 06:44:57 CST 2022`
- python version: `3.7.0 | packaged by conda-forge | (default, Nov 12 2018, 20:15:55)  [GCC 7.3.0]`
- espnet version: `espnet 202205`
- pytorch version: `pytorch 1.11.0`
- Git hash: ``
  - Commit date: ``

## asr_train_asr_streaming_conformer_rnn_transducer_raw_zh_char_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnnt_asr_model_valid.loss.ave_10best/dev|14326|14326|52.6|47.4|0.0|0.0|47.4|47.4|
|decode_asr_rnnt_asr_model_valid.loss.ave_10best/test|7176|7176|49.4|50.6|0.0|0.0|50.6|50.6|
|decode_asr_rnnt_lm_lm_train_lm_zh_char_valid.loss.ave_asr_model_valid.loss.ave_10best/dev|14326|14326|53.3|46.7|0.0|0.0|46.7|46.7|
|decode_asr_rnnt_lm_lm_train_lm_zh_char_valid.loss.ave_asr_model_valid.loss.ave_10best/test|7176|7176|50.6|49.4|0.0|0.0|49.4|49.4|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnnt_asr_model_valid.loss.ave_10best/dev|14326|205341|92.0|7.4|0.6|0.1|8.1|47.4|
|decode_asr_rnnt_asr_model_valid.loss.ave_10best/test|7176|104765|90.5|8.6|0.9|0.2|9.7|50.6|
|decode_asr_rnnt_lm_lm_train_lm_zh_char_valid.loss.ave_asr_model_valid.loss.ave_10best/dev|14326|205341|91.6|7.3|1.1|0.1|8.5|46.7|
|decode_asr_rnnt_lm_lm_train_lm_zh_char_valid.loss.ave_asr_model_valid.loss.ave_10best/test|7176|104765|90.2|8.2|1.5|0.1|9.9|49.4|

# Dynamic Chunked Conformer + RNNT RESULTS
## Environments
- date: `Thu Nov 17 19:24:07 CST 2022`
- python version: `3.7.0 | packaged by conda-forge | (default, Nov 12 2018, 20:15:55)  [GCC 7.3.0]`
- espnet version: `espnet 202209`
- pytorch version: `pytorch 1.11.0`
- Git hash: `209ffa0aea77757b9ee6882f06edfbba3aa261ee`
  - Commit date: `Fri Nov 11 10:24:24 2022 -0500`

## asr_train_asr_streaming_rnnt_raw_zh_char_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnnt_asr_model_valid.loss.ave_10best/dev|14326|14326|60.4|39.6|0.0|0.0|39.6|39.6|
|decode_asr_rnnt_asr_model_valid.loss.ave_10best/test|7176|7176|58.3|41.7|0.0|0.0|41.7|41.7|
|decode_asr_streaming_rnnt_asr_model_valid.loss.ave_10best/dev|14326|14326|29.5|70.5|0.0|0.0|70.5|70.5|
|decode_asr_streaming_rnnt_asr_model_valid.loss.ave_10best/test|7176|7176|25.9|74.1|0.0|0.0|74.1|74.1|
|decode_asr_streaming_rnnt_lm_lm_train_lm_rnn_zh_char_valid.loss.ave_asr_model_valid.loss.ave_10best/dev|14326|14326|28.6|71.4|0.0|0.0|71.4|71.4|
|decode_asr_streaming_rnnt_lm_lm_train_lm_rnn_zh_char_valid.loss.ave_asr_model_valid.loss.ave_10best/test|7176|7176|26.2|73.8|0.0|0.0|73.8|73.8|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnnt_asr_model_valid.loss.ave_10best/dev|14326|205341|94.6|5.2|0.2|0.1|5.5|39.6|
|decode_asr_rnnt_asr_model_valid.loss.ave_10best/test|7176|104765|94.0|5.7|0.3|0.1|6.0|41.7|
|decode_asr_streaming_rnnt_asr_model_valid.loss.ave_10best/dev|14326|205341|87.0|12.5|0.5|0.1|13.1|70.5|
|decode_asr_streaming_rnnt_asr_model_valid.loss.ave_10best/test|7176|104765|85.3|13.9|0.8|0.2|14.9|74.1|
|decode_asr_streaming_rnnt_lm_lm_train_lm_rnn_zh_char_valid.loss.ave_asr_model_valid.loss.ave_10best/dev|14326|205341|79.4|10.2|10.4|0.1|20.7|71.4|
|decode_asr_streaming_rnnt_lm_lm_train_lm_rnn_zh_char_valid.loss.ave_asr_model_valid.loss.ave_10best/test|7176|104765|77.7|10.7|11.6|0.1|22.4|73.8|

# Branchformer without sp RESULTS
## Environments
- date: `Wed Nov  9 17:58:28 CST 2022`
- python version: `3.7.0 | packaged by conda-forge | (default, Nov 12 2018, 20:15:55)  [GCC 7.3.0]`
- espnet version: `espnet 202205`
- pytorch version: `pytorch 1.11.0`
- Git hash: ``
  - Commit date: ``

## asr_train_asr_branchformer_raw_zh_char
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/dev|14326|14326|62.2|37.8|0.0|0.0|37.8|37.8|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/test|7176|7176|60.0|40.0|0.0|0.0|40.0|40.0|
|decode_asr_transformer_ngram_ngram_ngram_4gram_asr_model_valid.acc.ave_10best/dev|14326|14326|62.0|38.0|0.0|0.0|38.0|38.0|
|decode_asr_transformer_ngram_ngram_ngram_4gram_asr_model_valid.acc.ave_10best/test|7176|7176|59.8|40.2|0.0|0.0|40.2|40.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/dev|14326|205341|95.0|4.9|0.1|0.1|5.1|37.8|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/test|7176|104765|94.6|5.3|0.1|0.1|5.5|40.0|
|decode_asr_transformer_ngram_ngram_ngram_4gram_asr_model_valid.acc.ave_10best/dev|14326|205341|95.0|4.9|0.1|0.1|5.1|38.0|
|decode_asr_transformer_ngram_ngram_ngram_4gram_asr_model_valid.acc.ave_10best/test|7176|104765|94.5|5.3|0.2|0.1|5.6|40.2|

# Hubert(finetune) + Branchformer + Transformer RESULTS
## Environments
- date: `Tue Nov 22 11:54:20 CST 2022`
- python version: `3.7.0 | packaged by conda-forge | (default, Nov 12 2018, 20:15:55)  [GCC 7.3.0]`
- espnet version: `espnet 202205`
- pytorch version: `pytorch 1.11.0`
- Git hash: ``
  - Commit date: ``

## asr_train_asr_hubert_ft_branchformer_raw_zh_char
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/dev|14326|14326|62.0|38.0|0.0|0.0|38.0|38.0|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/test|7176|7176|59.6|40.4|0.0|0.0|40.4|40.4|
|decode_asr_transformer_ngram_ngram_ngram_4gram_asr_model_valid.acc.ave_10best/dev|14326|14326|61.9|38.1|0.0|0.0|38.1|38.1|
|decode_asr_transformer_ngram_ngram_ngram_4gram_asr_model_valid.acc.ave_10best/test|7176|7176|59.5|40.5|0.0|0.0|40.5|40.5|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/dev|14326|205341|95.1|4.8|0.1|0.1|5.0|38.0|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/test|7176|104765|94.7|5.1|0.1|0.1|5.4|40.4|
|decode_asr_transformer_ngram_ngram_ngram_4gram_asr_model_valid.acc.ave_10best/dev|14326|205341|95.1|4.8|0.1|0.1|5.0|38.1|
|decode_asr_transformer_ngram_ngram_ngram_4gram_asr_model_valid.acc.ave_10best/test|7176|104765|94.7|5.1|0.2|0.1|5.4|40.5|

# Hubert(freezed) + Branchformer + Transformer  RESULTS
## Environments
- date: `Thu Nov 10 17:17:51 CST 2022`
- python version: `3.7.0 | packaged by conda-forge | (default, Nov 12 2018, 20:15:55)  [GCC 7.3.0]`
- espnet version: `espnet 202205`
- pytorch version: `pytorch 1.11.0`
- Git hash: ``
  - Commit date: ``

## asr_train_asr_hubert_branchformer_raw_zh_char
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/dev|14326|14326|64.4|35.6|0.0|0.0|35.6|35.6|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/test|7176|7176|61.8|38.2|0.0|0.0|38.2|38.2|
|decode_asr_transformer_ngram_ngram_ngram_4gram_asr_model_valid.acc.ave_10best/dev|14326|14326|64.5|35.5|0.0|0.0|35.5|35.5|
|decode_asr_transformer_ngram_ngram_ngram_4gram_asr_model_valid.acc.ave_10best/test|7176|7176|62.0|38.0|0.0|0.0|38.0|38.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/dev|14326|205341|95.7|4.2|0.1|0.1|4.4|35.6|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/test|7176|104765|95.3|4.6|0.1|0.1|4.8|38.2|
|decode_asr_transformer_ngram_ngram_ngram_4gram_asr_model_valid.acc.ave_10best/dev|14326|205341|95.7|4.2|0.1|0.1|4.4|35.5|
|decode_asr_transformer_ngram_ngram_ngram_4gram_asr_model_valid.acc.ave_10best/test|7176|104765|95.3|4.6|0.1|0.1|4.7|38.0|

# Hubert encoder + Transformer decoder without sp RESULTS
## Environments
- date: `Wed Nov  9 19:19:22 CST 2022`
- python version: `3.7.0 | packaged by conda-forge | (default, Nov 12 2018, 20:15:55)  [GCC 7.3.0]`
- espnet version: `espnet 202205`
- pytorch version: `pytorch 1.11.0`
- Git hash: ``
  - Commit date: ``

## asr_train_asr_hubert_base_raw_zh_char
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/dev|14326|14326|65.5|34.5|0.0|0.0|34.5|34.5|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/test|7176|7176|62.5|37.5|0.0|0.0|37.5|37.5|
|decode_asr_transformer_ngram_lm_lm_train_lm_zh_char_valid.loss.ave_ngram_ngram_3gram_asr_model_valid.acc.best/dev|14326|14326|59.1|40.9|0.0|0.0|40.9|40.9|
|decode_asr_transformer_ngram_lm_lm_train_lm_zh_char_valid.loss.ave_ngram_ngram_3gram_asr_model_valid.acc.best/test|7176|7176|57.4|42.6|0.0|0.0|42.6|42.6|
|decode_asr_transformer_ngram_ngram_ngram_3gram_asr_model_valid.acc.best/dev|14326|14326|63.4|36.6|0.0|0.0|36.6|36.6|
|decode_asr_transformer_ngram_ngram_ngram_3gram_asr_model_valid.acc.best/test|7176|7176|61.1|38.9|0.0|0.0|38.9|38.9|
|decode_asr_transformer_ngram_ngram_ngram_4gram_asr_model_valid.acc.ave_10best/dev|14326|14326|65.3|34.7|0.0|0.0|34.7|34.7|
|decode_asr_transformer_ngram_ngram_ngram_4gram_asr_model_valid.acc.ave_10best/test|7176|7176|62.1|37.9|0.0|0.0|37.9|37.9|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/dev|14326|205341|95.8|4.1|0.1|0.1|4.3|34.5|
|decode_asr_transformer_ngram_asr_model_valid.acc.ave_10best/test|7176|104765|95.3|4.6|0.1|0.1|4.8|37.5|
|decode_asr_transformer_ngram_lm_lm_train_lm_zh_char_valid.loss.ave_ngram_ngram_3gram_asr_model_valid.acc.best/dev|14326|205341|94.5|4.9|0.6|0.1|5.6|40.9|
|decode_asr_transformer_ngram_lm_lm_train_lm_zh_char_valid.loss.ave_ngram_ngram_3gram_asr_model_valid.acc.best/test|7176|104765|94.1|5.2|0.7|0.1|6.0|42.6|
|decode_asr_transformer_ngram_ngram_ngram_3gram_asr_model_valid.acc.best/dev|14326|205341|95.5|4.4|0.1|0.1|4.6|36.6|
|decode_asr_transformer_ngram_ngram_ngram_3gram_asr_model_valid.acc.best/test|7176|104765|95.0|4.8|0.2|0.1|5.0|38.9|
|decode_asr_transformer_ngram_ngram_ngram_4gram_asr_model_valid.acc.ave_10best/dev|14326|205341|95.8|4.1|0.1|0.1|4.3|34.7|
|decode_asr_transformer_ngram_ngram_ngram_4gram_asr_model_valid.acc.ave_10best/test|7176|104765|95.3|4.6|0.2|0.0|4.8|37.9|


# Streaming Conformer + specaug + speed perturbation: feats=raw, n_fft=512, hop_length=128
## Environments
- date: `Mon Aug 23 16:31:48 CST 2021`
- python version: `3.7.9 (default, Aug 31 2020, 12:42:55)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.9`
- pytorch version: `pytorch 1.5.0`
- Git hash: `b94d07028099a80c9c690341981ae7d550b5ca24`
  - Commit date: `Mon Aug 23 00:47:47 2021 +0800`

## With Transformer LM
- Model link: (wait for upload)
- ASR config: [./conf/train_asr_streaming_cpnformer.yaml](./conf/train_asr_streaming_conformer.yaml)
- LM config: [./conf/tuning/train_lm_transformer.yaml](./conf/tuning/train_lm_transformer.yaml)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_streaming_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|94.0|5.8|0.3|0.3|6.3|42.2|
|decode_asr_streaming_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|92.9|6.7|0.5|0.7|7.8|46.2|
# Streaming Transformer + speed perturbation: feats=raw, n_fft=512, hop_length=128
## Environments
- date: `Tue Aug 17 01:20:32 CST 2021`
- python version: `3.7.9 (default, Aug 31 2020, 12:42:55)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.9`
- pytorch version: `pytorch 1.5.0`
- Git hash: `6f5f848e0a9bfca1b73393779233bde34add3df1`
  - Commit date: `Mon Aug 16 21:50:08 2021 +0800`

## With Transformer LM
- Model link: (wait for upload)
- ASR config: [./conf/train_asr_streaming_transformer.yaml](./conf/train_asr_streaming_transformer.yaml)
- LM config: [./conf/tuning/train_lm_transformer.yaml](./conf/tuning/train_lm_transformer.yaml)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_streaming_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|93.6|6.2|0.1|0.5|6.8|46.8|
|decode_asr_streaming_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|93.0|6.7|0.2|0.8|7.8|50.7|

# Conformer + specaug + speed perturbation: feats=raw, n_fft=512, hop_length=128
## Environments
- date: `Fri Oct 16 11:10:17 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.0`
- pytorch version: `pytorch 1.6.0`
- Git hash: `20b0c89369d9dd3e05780b65fdd00a9b4f4891e5`
  - Commit date: `Mon Oct 12 09:28:20 2020 -0400`

## With Transformer LM
- Model link: https://zenodo.org/record/4105763#.X40xe2j7QUE
- ASR config: [./conf/tuning/train_asr_conformer.yaml](./conf/tuning/train_asr_conformer.yaml)
- LM config: [./conf/tuning/train_lm_transformer.yaml](./conf/tuning/train_lm_transformer.yaml)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnn_lm_lm_train_lm_transformer_char_batch_bins2000000_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|95.7|4.2|0.1|0.1|4.4|33.7|
|decode_asr_rnn_lm_lm_train_lm_transformer_char_batch_bins2000000_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|95.4|4.5|0.1|0.1|4.7|35.0|

## With RNN LM
- ASR config: [./conf/tuning/train_asr_conformer.yaml](./conf/tuning/train_asr_conformer.yaml)
- LM config: [./conf/tuning/train_lm_rnn2.yaml](./conf/tuning/train_lm_rnn2.yaml)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|95.5|4.4|0.1|0.1|4.6|35.2|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|95.2|4.7|0.1|0.1|4.9|36.5|

## Without LM
- ASR config: [./conf/tuning/train_asr_conformer.yaml](./conf/tuning/train_asr_conformer.yaml)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnn_asr_model_valid.acc.ave/dev|14326|205341|95.6|4.3|0.1|0.1|4.5|35.0|
|decode_asr_rnn_asr_model_valid.acc.ave/test|7176|104765|95.2|4.7|0.1|0.1|4.9|36.7|

# Transformer + speed perturbation: feats=raw with same LM with the privious setting

I compared between `n_fft=512, hop_length=128`, `n_fft=400, hop_length=160`,  and `n_fft=512, hop_length=256`
with searching the best `batch_bins` to get the suitable configuration for each hop_length.

## Environments
- date: `Fri Oct 16 11:10:17 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.0`
- pytorch version: `pytorch 1.6.0`
- Git hash: `20b0c89369d9dd3e05780b65fdd00a9b4f4891e5`
  - Commit date: `Mon Oct 12 09:28:20 2020 -0400`

## n_fft=512, hop_length=128
asr_train_asr_transformer2_raw_char_batch_typenumel_batch_bins8500000_optim_conflr0.0005_scheduler_confwarmup_steps30000_sp

- ASR config: [./conf/tuning/train_asr_transformer3.yaml](./conf/tuning/train_asr_transformer3.yaml)
- LM config: [./conf/tuning/train_lm_rnn2.yaml](./conf/tuning/train_lm_rnn2.yaml)


### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|94.2|5.7|0.1|0.1|5.9|42.6|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|93.7|6.1|0.2|0.1|6.4|45.0|


## n_fft=400, hop_length=160
asr_train_asr_transformer2_raw_char_frontend_confn_fft400_frontend_confhop_length160_batch_typenumel_batch_bins6500000_optim_conflr0.0005_scheduler_confwarmup_steps30000_sp

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|94.1|5.7|0.1|0.1|6.0|43.0|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|93.5|6.3|0.2|0.1|6.6|45.4|

## n_fft=512, hop_length=256
asr_train_asr_transformer2_raw_char_frontend_confn_fft512_frontend_confhop_length256_batch_typenumel_batch_bins6000000_sp

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|94.0|5.9|0.1|0.1|6.1|43.5|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|93.3|6.5|0.2|0.1|6.8|45.8|


# Transformer + speed perturbation: feats=fbank_pitch, RNN-LM
Compatible setting with espnet1 to reproduce the previou result

## Environments
- date: `Fri Oct 16 11:10:17 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.0`
- pytorch version: `pytorch 1.6.0`
- Git hash: `20b0c89369d9dd3e05780b65fdd00a9b4f4891e5`
  - Commit date: `Mon Oct 12 09:28:20 2020 -0400`

- ASR config: [./conf/tuning/train_asr_transformer2.yaml](./conf/tuning/train_asr_transformer2.yaml)
- LM config: [./conf/tuning/train_lm_rnn2.yaml](./conf/tuning/train_lm_rnn2.yaml)

## asr_train_asr_transformer2_fbank_pitch_char_sp
### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|94.0|5.9|0.1|0.1|6.1|43.4|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|93.4|6.4|0.2|0.1|6.7|45.9|

# The first result
## Environments
- date: `Sun Feb  2 02:03:55 CST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.6.0`
- pytorch version: `pytorch 1.1.0`
- Git hash: `e0fd073a70bcded6a0e6a3587630410a994ccdb8` (+ fixing https://github.com/espnet/espnet/pull/1533)
  - Commit date: `Sat Jan 11 06:09:24 2020 +0900`

## asr_train_asr_rnn_new_fbank_pitch_char
### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_devdecode_asr_rnn_lm_valid.loss.best_asr_model_valid.acc.best|14326|205341|92.6|7.2|0.2|0.1|7.5|49.6|
|decode_testdecode_asr_rnn_lm_valid.loss.best_asr_model_valid.acc.best|7176|104765|91.6|8.2|0.3|0.2|8.6|53.4|

## asr_train_asr_transformer_lr0.002_fbank_pitch_char
### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_decode_asr_rnn_lm_train_lm_char_valid.loss.best_asr_model_valid.acc.best|14326|205341|93.3|6.5|0.2|0.1|6.8|45.6|
|decode_test_decode_asr_rnn_lm_train_lm_char_valid.loss.best_asr_model_valid.acc.best|7176|104765|92.7|7.1|0.3|0.1|7.4|47.6|

