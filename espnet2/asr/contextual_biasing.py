"""
Contextual Biasing as Rescorer

"""

import torch
import json
import logging

from espnet.nets.rescorer_interface import BatchRescorerInterface
from espnet.asr.asr_utils import char_score
from espnet.utils.pytrie import complete_match, query_match_results, Trie

from abc import ABC

class BiasingState(ABC):
    def __init__(self):
        self.accumulate_score = 0.0
        self.prefix_ids = []
        self.prefix_start_id = 1

    def reset(self, prefix_start_id=None):
        self.__init__()
        if prefix_start_id is not None:
            self.prefix_start_id = prefix_start_id


class ContextualBiasingRescorer(BatchRescorerInterface):
    """ContextualBiasing(A.K.A hot words boosting) implemented throught RescorerInterface."""

    def __init__(self, token_list, tokenizer, biasing_words_file,
                 dict_frequency_file, min_boost_score=1.0, max_boost_score=10.0):
        """Initialize ContextBiasingBase.
        Args:
            ngram_model: ngram model path
            token_list: token list from dict or model.json
        """

        self.token_list = token_list
        with open(biasing_words_file, "r") as f:
            token2idx = {token: idx for idx, token in enumerate(token_list)}
            biasing_words = [l.strip() for l in f.readlines()]
            biasing_words_idx = []
            biasing_words_in = []

            for word in biasing_words:
                word_idx = []
                tokens = tokenizer.text2tokens(word)
                for token in tokens:
                    idx = token2idx.get(token, -1)
                    if idx < 0:
                        logging.warning("ignore biasing word {}, {} was not in dict".format(word, token))
                        word_idx = None
                        break
                    word_idx.append(idx)
                if word_idx:
                    biasing_words_idx.append(word_idx)
                    biasing_words_in.append(word)
            logging.info("number of biasing words: {}".format(len(biasing_words_idx)))

            biasing_words_trie = Trie().fromkeys(biasing_words_idx, value=1)
            self.trie = biasing_words_trie
            biasing_words = biasing_words_in

        with open(dict_frequency_file, "r") as freq_file:
            token_freq = json.load(freq_file)
            token2score = char_score(token_freq, min_score=min_boost_score, max_score=max_boost_score)
            biasing_tokens = set([token for word in biasing_words for token in tokenizer.text2tokens(word)])
            idx2score = {token2idx[token]: score for token, score in token2score.items() if token in biasing_tokens}
            self.idx2score = idx2score

        self.query_cache = {}

    def init_state(self, x):
        """Initialize tmp state."""
        state = BiasingState()
        return state

    def rescore(self, hyp, state):
        """Score interface for both full and partial scorer.
        Args:
            y: previous token
            next_token: next token need to be score
            state: previous state
        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.
        """
        bias_score = 0.0
        yseq = hyp
        yseq_len = len(yseq)
        prefix_start_id = state.prefix_start_id
        # skip eos
        if prefix_start_id >= yseq_len or yseq[-1] == self.token_list[-1]:
            return bias_score, state
        prefix_ids = yseq[prefix_start_id:]
        if prefix_ids in self.query_cache:
            prefix_match = self.query_cache[prefix_ids]
        else:
            prefix_match = self.trie.items(prefix=prefix_ids)
            self.query_cache[prefix_ids] = prefix_match
        prefix_words = [self.token_list[idx] for idx in prefix_ids]
        if prefix_match:
            token_score = self.idx2score[prefix_ids]
            state.accumulate_score += token_score
            bias_score += token_score
            if complete_match(prefix_match, prefix_ids):
                state.reset(prefix_start_id=yseq_len)
        else:
            # subtract accumulated biasing score due to partial match
            if len(prefix_ids) > 1:
                bias_score -= state.accumulate_score
                remained_seq = yseq[prefix_start_id + 1:]
                remained_start_idx, partial_score = query_match_results(self.trie, remained_seq, self.idx2score)
                if remained_start_idx is None:
                    state.reset(prefix_start_id=yseq_len)
                else:
                    bias_score += partial_score
                    state.accumulate_score = partial_score
                    state.prefix_start_id += remained_start_idx + 1
            else:
                state.reset(prefix_start_id=yseq_len)

        return torch.tensor(bias_score).to(yseq), state

