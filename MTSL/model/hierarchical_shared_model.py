import torch
import torch.nn as nn
from chaincrf import ChainCRF
from torch.nn import Embedding
import utils
import torch.nn.utils.rnn as rnn_utils
from allennlp.modules.elmo import Elmo


class HierarchicalSharedModel(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_labels, num_filters,
                 kernel_size, rnn_mode, hidden_size, num_layers, embedd_word=None, p_in=0.33, p_out=0.5,
                 p_rnn=(0.5, 0.5), bigram=True, use_crf=True, use_lm=True, use_elmo=False):
        super(HierarchicalSharedModel, self).__init__()
        self.use_elmo = use_elmo
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
        self.rnn_1 = RNN(word_dim + num_filters, hidden_size, num_layers=num_layers, batch_first=True,
                         bidirectional=True, dropout=p_rnn[1])
        self.rnn_2 = RNN(word_dim + num_filters + hidden_size*2, hidden_size, num_layers=num_layers, batch_first=True,
                         bidirectional=True, dropout=p_rnn[1])
        # self.rnn_3 = RNN(word_dim + num_filters + hidden_size*2, hidden_size, num_layers=num_layers, batch_first=True,
        #                  bidirectional=True, dropout=p_rnn[1])
        if self.use_crf:
            self.crf_1 = ChainCRF(hidden_size * 2, num_labels[0], bigram=bigram)
            self.crf_2 = ChainCRF(hidden_size * 2, num_labels[1], bigram=bigram)
            # self.crf_3 = ChainCRF(hidden_size * 2, num_labels[2], bigram=bigram)
        else:
            self.dense_softmax_1 = nn.Linear(hidden_size * 2, num_labels[0])
            self.dense_softmax_2 = nn.Linear(hidden_size * 2, num_labels[1])
            # self.dense_softmax_3 = nn.Linear(hidden_size * 2, num_labels[2])
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.nll_loss = nn.NLLLoss(size_average=False, reduce=False)
        if self.use_lm:
            self.dense_fw_1 = nn.Linear(hidden_size, num_words)
            self.dense_bw_1 = nn.Linear(hidden_size, num_words)
            self.dense_fw_2 = nn.Linear(hidden_size, num_words)
            self.dense_bw_2 = nn.Linear(hidden_size, num_words)
            if self.use_crf:
                self.logsoftmax = nn.LogSoftmax(dim=1)
                self.nll_loss = nn.NLLLoss(size_average=False, reduce=False)

    def _get_rnn_output(self, input_word, input_char, task, mask=None, length=None, hx=None):
        # hack length from mask
        # we do not hack mask from length for special reasons.
        # Thus, always provide mask if it is necessary.
        if length is None and mask is not None:
            length = mask.data.sum(dim=1).long()
        if self.use_elmo:
            input = self.elmo(input_word)
            # mask = input['mask']
            # mask = mask.float()
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
        if length is not None:
            seq_input, hx, rev_order, mask, lens = utils.prepare_rnn_seq(input, length, hx=hx, masks=mask, batch_first=True)
            if task in ['chunk', 'ner', 'pos', 'ontonotes', 'fgner']:
                seq_output, hn = self.rnn_1(seq_input, hx=hx)
                seq_input, _ = rnn_utils.pad_packed_sequence(seq_input, batch_first=True)
            if task in ['fgner']:
                seq_output, _ = rnn_utils.pad_packed_sequence(seq_output, batch_first=True)
                output_size = seq_output.size()
                seq_output = seq_output.view(output_size[0], output_size[1], 2, -1)
                hidden_fw = seq_output[:, :, 0]
                hidden_bw = seq_output[:, :, 1]
                hidden_input = torch.cat((seq_input, hidden_fw, hidden_bw), 2)
                hidden_input = rnn_utils.pack_padded_sequence(hidden_input, lens, batch_first=True)
                seq_output, hn = self.rnn_2(hidden_input, hx=hx)
            # if task in ['fgner']:
            #     seq_output, _ = rnn_utils.pad_packed_sequence(seq_output, batch_first=True)
            #     output_size = seq_output.size()
            #     seq_output = seq_output.view(output_size[0], output_size[1], 2, -1)
            #     hidden_fw = seq_output[:, :, 0]
            #     hidden_bw = seq_output[:, :, 1]
            #     hidden_input = torch.cat((seq_input, hidden_fw, hidden_bw), 2)
            #     hidden_input = rnn_utils.pack_padded_sequence(hidden_input, lens, batch_first=True)
            #     seq_output, hn = self.rnn_3(hidden_input, hx=hx)
            output, hn = utils.recover_rnn_seq(seq_output, rev_order, hx=hn, batch_first=True)
        output = self.dropout_out(output)
        if self.use_lm:
            output_size = output.size()
            output_lm = output.view(output_size[0], output_size[1], 2, -1)
            output_fw = output_lm[:, :, 0]
            output_bw = output_lm[:, :, 1]
            return output, hn, mask, length, output_fw, output_bw
        else:
            return output, hn, mask, length

    def forward(self, input_word, input_char, mask=None, length=None, hx=None):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(input_word, input_char, mask=mask, length=length, hx=hx)
        # [batch, length, num_label,  num_label]
        return self.crf(output, mask=mask), mask

    def loss(self, input_word, input_char, target, task, target_fw, target_bw, mask=None, length=None, hx=None, leading_symbolic=0):
        # output from rnn [batch, length, tag_space]
        if self.use_lm:
            output, _, mask, length, output_fw, output_bw = self._get_rnn_output(input_word, input_char, task, mask=mask, length=length, hx=hx)
            if task == 'fgner':
                output_fw = self.dense_fw_2(output_fw)
                output_bw = self.dense_bw_2(output_bw)
            else:
                output_fw = self.dense_fw_1(output_fw)
                output_bw = self.dense_bw_1(output_bw)
            output_size = output_fw.size()
            output_size = (output_size[0] * output_size[1], output_size[2])
            output_fw = output_fw.view(output_size)
            output_bw = output_bw.view(output_size)
        else:
            output, _, mask, length = self._get_rnn_output(input_word, input_char, task, mask=mask, length=length, hx=hx)
        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]
            if self.use_lm:
                target_fw = target_fw[:, :max_len].contiguous()
                target_bw = target_bw[:, :max_len].contiguous()
        # [batch, length, num_label,  num_label]
        if self.use_crf:
            if self.use_lm:
                fw_loss = (self.nll_loss(self.logsoftmax(output_fw), target_fw.view(-1)) * mask.contiguous().view(
                    -1)).sum() / mask.sum()
                bw_loss = (self.nll_loss(self.logsoftmax(output_bw), target_bw.view(-1)) * mask.contiguous().view(
                    -1)).sum() / mask.sum()
                if task in ['chunk', 'ner', 'pos', 'ontonotes']:
                    return self.crf_1.loss(output, target, mask=mask).mean() + 0.05 * (fw_loss + bw_loss)
                elif task == 'fgner':
                    return self.crf_2.loss(output, target, mask=mask).mean() + 0.05 * (fw_loss + bw_loss)
                # elif task == 'fgner':
                #     return self.crf_3.loss(output, target, mask=mask).mean() + 0.05 * (fw_loss + bw_loss)
            else:
                if task == 'source':
                    return self.crf_1.loss(output, target, mask=mask).mean()
                elif task == 'target':
                    return self.crf_2.loss(output, target, mask=mask).mean()
                # elif task == 'fgner':
                #     return self.crf_3.loss(output, target, mask=mask).mean()
        else:
            target = target.contiguous()
            if task in ['chunk', 'ner', 'pos', 'ontonotes']:
                output = self.dense_softmax_1(output)
            elif task == 'fgner':
                output = self.dense_softmax_2(output)
            # elif task == 'fgner':
            #     output = self.dense_softmax_3(output)
            # preds = [batch, length]
            _, preds = torch.max(output[:, :, leading_symbolic:], dim=2)
            preds += leading_symbolic
            output_size = output.size()
            # [batch * length, num_labels]
            output_size = (output_size[0] * output_size[1], output_size[2])
            output = output.view(output_size)
            if self.use_lm:
                fw_loss = (self.nll_loss(self.logsoftmax(output_fw), target_fw.view(-1)) * mask.contiguous().view(
                    -1)).sum() / mask.sum()
                bw_loss = (self.nll_loss(self.logsoftmax(output_bw), target_bw.view(-1)) * mask.contiguous().view(
                    -1)).sum() / mask.sum()
                return (self.nll_loss(self.logsoftmax(output), target.view(-1)) * mask.contiguous().view(-1)).sum() + 0.05 * (fw_loss + bw_loss) / mask.sum(), preds
            else:
                return (self.nll_loss(self.logsoftmax(output), target.view(-1)) * mask.contiguous().view(-1)).sum() / mask.sum(), preds

    def decode(self, input_word, input_char, task, target=None, mask=None, length=None, hx=None, leading_symbolic=0):
        # output from rnn [batch, length, tag_space]
        if self.use_lm:
            output, _, mask, length, output_fw, output_bw = self._get_rnn_output(input_word, input_char, task, mask=mask, length=length, hx=hx)
        else:
            output, _, mask, length,  = self._get_rnn_output(input_word, input_char, task, mask=mask, length=length, hx=hx)
        if target is None:
            if task in ['chunk', 'ner', 'pos', 'ontonotes']:
                return self.crf_1.decode(output, mask=mask, leading_symbolic=leading_symbolic), None
            elif task == 'fgner':
                return self.crf_2.decode(output, mask=mask, leading_symbolic=leading_symbolic), None
            # elif task == 'fgner':
            #     return self.crf_3.decode(output, mask=mask, leading_symbolic=leading_symbolic), None
        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]
        if task in ['chunk', 'ner', 'pos', 'ontonotes']:
            preds = self.crf_1.decode(output, mask=mask, leading_symbolic=leading_symbolic)
        elif task == 'fgner':
            preds = self.crf_2.decode(output, mask=mask, leading_symbolic=leading_symbolic)
        # elif task == 'fgner':
        #     preds = self.crf_3.decode(output, mask=mask, leading_symbolic=leading_symbolic)
        if mask is None:
            return preds, torch.eq(preds, target.data).float().sum()
        else:
            return preds, (torch.eq(preds, target.data).float() * mask.data).sum()

