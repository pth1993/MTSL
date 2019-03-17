import torch
import torch.nn as nn
from chaincrf import ChainCRF
from torch.nn import Embedding
from allennlp.modules.elmo import Elmo
import utils


class RNNSharedModel(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_labels, num_filters,
                 kernel_size, rnn_mode, hidden_size, num_layers, embedd_word=None, p_in=0.33, p_out=0.5,
                 p_rnn=(0.5, 0.5), lm_loss=0.05, bigram=True, use_crf=True, use_lm=True, use_elmo=False,
                 lm_mode='unshared'):
        super(RNNSharedModel, self).__init__()
        self.lm_loss = lm_loss
        self.use_elmo = use_elmo
        self.lm_mode = lm_mode
        if self.use_elmo:
            option_file, weight_file = embedd_word
            self.elmo = Elmo(option_file, weight_file, 2, dropout=0)
            word_dim = 1024
            num_filters = 0
        else:
            if isinstance(embedd_word, torch.Tensor):
                self.word_embedd = Embedding.from_pretrained(embedd_word, freeze=False)
            else:
                self.word_embedd = Embedding(num_words, word_dim)
            self.char_embedd = Embedding(num_chars, char_dim)
            self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1)
            self.dropout_in = nn.Dropout(p=p_in)
        self.dropout_rnn_in = nn.Dropout(p=p_rnn[0])
        self.dropout_out = nn.Dropout(p_out)
        self.use_crf = use_crf
        self.use_lm = use_lm
        if rnn_mode == 'RNN':
            RNN = nn.RNN
        elif rnn_mode == 'LSTM':
            RNN = nn.LSTM
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)
        self.rnn = RNN(word_dim + num_filters, hidden_size, num_layers=num_layers, batch_first=True,
                       bidirectional=True, dropout=p_rnn[1])
        if self.use_crf:
            self.crf_1 = ChainCRF(hidden_size * 2, num_labels[0], bigram=bigram)
            self.crf_2 = ChainCRF(hidden_size * 2, num_labels[1], bigram=bigram)
        else:
            self.dense_softmax_1 = nn.Linear(hidden_size * 2, num_labels[0])
            self.dense_softmax_2 = nn.Linear(hidden_size * 2, num_labels[1])
        if self.use_lm:
            if self.lm_mode == 'unshared':
                self.dense_fw_1 = nn.Linear(hidden_size, num_words)
                self.dense_bw_1 = nn.Linear(hidden_size, num_words)
                self.dense_fw_2 = nn.Linear(hidden_size, num_words)
                self.dense_bw_2 = nn.Linear(hidden_size, num_words)
            elif self.lm_mode == 'shared':
                self.dense_fw = nn.Linear(hidden_size, num_words)
                self.dense_bw = nn.Linear(hidden_size, num_words)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss(size_average=False, reduce=False)

    def _get_rnn_output(self, input_word, input_char, mask, hx=None):
        length = mask.data.sum(dim=1).long()
        # [batch, length, word_dim]
        if self.use_elmo:
            input = self.elmo(input_word)
            input = input['elmo_representations'][1]
        else:
            # [batch, length, word_dim]
            word = self.word_embedd(input_word)
            # [batch, length, char_length, char_dim]
            char = self.char_embedd(input_char)
            char_size = char.size()
            # first transform to [batch *length, char_length, char_dim]
            # then transpose to [batch * length, char_dim, char_length]
            char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
            # put into cnn [batch*length, char_filters, char_length]
            # then put into maxpooling [batch * length, char_filters]
            char, _ = self.conv1d(char).max(dim=2)
            # reshape to [batch, length, char_filters]
            char = torch.tanh(char).view(char_size[0], char_size[1], -1)
            # apply dropout word on input
            word = self.dropout_in(word)
            char = self.dropout_in(char)
            # concatenate word and char [batch, length, word_dim+char_filter]
            input = torch.cat([word, char], dim=2)
        # apply dropout
        input = self.dropout_rnn_in(input)
        # prepare packed_sequence
        seq_input, hx, rev_order, mask, _ = utils.prepare_rnn_seq(input, length, hx=hx, masks=mask, batch_first=True)
        seq_output, hn = self.rnn(seq_input, hx=hx)
        output, hn = utils.recover_rnn_seq(seq_output, rev_order, hx=hn, batch_first=True)
        output = self.dropout_out(output)
        if self.use_lm:
            output_size = output.size()
            # print output_size
            lm = output.view(output_size[0], output_size[1], 2, -1)
            # print output_lm.size()
            lm_fw = lm[:, :, 0]
            lm_bw = lm[:, :, 1]
            return output, hn, mask, length, lm_fw, lm_bw
        else:
            return output, hn, mask, length

    def forward(self, input_word, input_char, mask, hx=None):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(input_word, input_char, mask, hx=hx)
        # [batch, length, num_label,  num_label]
        return self.crf(output, mask=mask), mask

    def loss(self, input_word, input_char, target, main_task, target_fw, target_bw, mask, hx=None, leading_symbolic=0):
        # output from rnn [batch, length, tag_space]
        if self.use_lm:
            output, _, mask, length, lm_fw, lm_bw = self._get_rnn_output(input_word, input_char, mask, hx=hx)
            if self.lm_mode == 'unshared':
                if main_task:
                    lm_fw = self.dense_fw_2(lm_fw)
                    lm_bw = self.dense_bw_2(lm_bw)
                else:
                    lm_fw = self.dense_fw_1(lm_fw)
                    lm_bw = self.dense_bw_1(lm_bw)
            elif self.lm_mode == 'shared':
                lm_fw = self.dense_fw(lm_fw)
                lm_bw = self.dense_bw(lm_bw)
            else:
                raise ValueError('Unknown LM mode: %s' % self.lm_mode)
            output_size = lm_fw.size()
            output_size = (output_size[0] * output_size[1], output_size[2])
            lm_fw = lm_fw.view(output_size)
            lm_bw = lm_bw.view(output_size)
            max_len = length.max()
            target_fw = target_fw[:, :max_len].contiguous()
            target_bw = target_bw[:, :max_len].contiguous()
            fw_loss = (self.nll_loss(self.logsoftmax(lm_fw), target_fw.view(-1)) * mask.contiguous().view(
                -1)).sum() / mask.sum()
            bw_loss = (self.nll_loss(self.logsoftmax(lm_bw), target_bw.view(-1)) * mask.contiguous().view(
                -1)).sum() / mask.sum()
        else:
            output, _, mask, length = self._get_rnn_output(input_word, input_char, mask, hx=hx)
            max_len = length.max()
        target = target[:, :max_len]
        # [batch, length, num_label,  num_label]
        if self.use_crf:
            if self.use_lm:
                if main_task:
                    return self.crf_2.loss(output, target, mask=mask).mean() + self.lm_loss * (fw_loss + bw_loss)
                else:
                    return self.crf_1.loss(output, target, mask=mask).mean() + self.lm_loss * (fw_loss + bw_loss)
            else:
                if main_task:
                    return self.crf_2.loss(output, target, mask=mask).mean()
                else:
                    return self.crf_1.loss(output, target, mask=mask).mean()
        else:
            target = target.contiguous()
            if main_task:
                output = self.dense_softmax_2(output)
            else:
                output = self.dense_softmax_1(output)
            # preds = [batch, length]
            _, preds = torch.max(output[:, :, leading_symbolic:], dim=2)
            preds += leading_symbolic
            output_size = output.size()
            # [batch * length, num_labels]
            output_size = (output_size[0] * output_size[1], output_size[2])
            output = output.view(output_size)
            if self.use_lm:
                return (self.nll_loss(self.logsoftmax(output), target.view(-1)) * mask.contiguous().view(-1)).sum() / \
                       mask.sum() + self.lm_loss * (fw_loss + bw_loss), preds
            else:
                return (self.nll_loss(self.logsoftmax(output), target.view(-1)) * mask.contiguous().view(-1)).sum() / \
                       mask.sum(), preds

    def decode(self, input_word, input_char, target, main_task, mask, hx=None, leading_symbolic=0):
        # output from rnn [batch, length, tag_space]
        if self.use_lm:
            output, _, mask, length, lm_fw, lm_bw = self._get_rnn_output(input_word, input_char, mask, hx=hx)
        else:
            output, _, mask, length,  = self._get_rnn_output(input_word, input_char, mask, hx=hx)
        max_len = length.max()
        target = target[:, :max_len]
        if main_task:
            preds = self.crf_2.decode(output, mask=mask, leading_symbolic=leading_symbolic)
        else:
            preds = self.crf_1.decode(output, mask=mask, leading_symbolic=leading_symbolic)
        return preds, (torch.eq(preds, target.data).float() * mask.data).sum()

