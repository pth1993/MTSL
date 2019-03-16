import os.path
import subprocess
import shlex
from word2index import Word2Index
from datareader import DataReader
from constant import DIGIT_RE, MAX_CHAR_LENGTH, NUM_CHAR_PAD, PAD_WORD, PAD_START, PAD_END, PAD_LABEL, PAD_CHAR, \
    UNK_ID, PAD_ID_WORD, PAD_ID_CHAR, PAD_ID_TAG
from logger import get_logger
import codecs
import numpy as np
import torch
from allennlp.modules.elmo import batch_to_ids

# Special vocabulary symbols - we always put them at the start.
_START_VOCAB = [PAD_WORD, PAD_START, PAD_END]


def construct_word_embedding_table(embedd_dict, embedd_dim, word_word2index):
    scale = np.sqrt(3.0 / embedd_dim)
    table = np.empty([word_word2index.size(), embedd_dim], dtype=np.float32)
    table[UNK_ID, :] = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
    oov = 0
    for word, index in word_word2index.items():
        if word in embedd_dict:
            embedding = embedd_dict[word]
        elif word.lower() in embedd_dict:
            embedding = embedd_dict[word.lower()]
        else:
            embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
            oov += 1
        table[index, :] = embedding
    print('oov: %d' % oov)
    return torch.from_numpy(table)


def create_word2indexs(word2index_directory, train_path, label_type, test_paths=None, max_vocabulary_size=60000,
                       embedd_dict=None, min_occurence=1, normalize_digits=True):
    def expand_vocab():
        vocab_set = set(vocab_list)
        for i in range(len(test_paths)):
            for data_path in test_paths[i]:
                with codecs.open(data_path, 'r', 'utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if len(line) == 0:
                            continue
                        tokens = line.split('\t')
                        word = DIGIT_RE.sub("0", tokens[0]) if normalize_digits else tokens[0]
                        label = tokens[1]
                        label_word2index_list[i].add(label)
                        if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                            vocab_set.add(word)
                            vocab_list.append(word)

    logger = get_logger("Create Word2Indexs")
    word_word2index = Word2Index('word', default_value=True, singleton=True)
    char_word2index = Word2Index('character', default_value=True)
    label_word2index_list = [Word2Index('label_' + label_type[i]) for i in range(len(train_path))]
    if not os.path.isdir(word2index_directory):
        logger.info("Creating Word2Indexs: %s" % word2index_directory)
        char_word2index.add(PAD_CHAR)
        vocab = dict()
        for i in range(len(train_path)):
            label_word2index_list[i].add(PAD_LABEL)
            with codecs.open(train_path[i], 'r', 'utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    tokens = line.split('\t')
                    for char in tokens[0]:
                        char_word2index.add(char)
                    word = DIGIT_RE.sub("0", tokens[0]) if normalize_digits else tokens[0]
                    label = tokens[1]
                    label_word2index_list[i].add(label)
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
        # collect singletons
        singletons = set([word for word, count in vocab.items() if count <= min_occurence])
        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        if embedd_dict is not None:
            for word in vocab.keys():
                if word in embedd_dict or word.lower() in embedd_dict:
                    vocab[word] += min_occurence
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        logger.info("Total Vocabulary Size: %d" % len(vocab_list))
        logger.info("Total Singleton Size:  %d" % len(singletons))
        vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurence]
        logger.info("Total Vocabulary Size (w.o rare words): %d" % len(vocab_list))
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        if test_paths is not None and embedd_dict is not None:
            expand_vocab()
        for word in vocab_list:
            word_word2index.add(word)
            if word in singletons:
                word_word2index.add_singleton(word_word2index.get_index(word))
        word_word2index.save(word2index_directory)
        char_word2index.save(word2index_directory)
        for i in range(len(train_path)):
            label_word2index_list[i].save(word2index_directory)
    else:
        word_word2index.load(word2index_directory)
        char_word2index.load(word2index_directory)
        for i in range(len(train_path)):
            label_word2index_list[i].load(word2index_directory)
    word_word2index.close()
    char_word2index.close()
    for i in range(len(train_path)):
        label_word2index_list[i].close()
    logger.info("Word Alphabet Size (Singleton): %d (%d)" % (word_word2index.size(), word_word2index.singleton_size()))
    logger.info("Character Alphabet Size: %d" % char_word2index.size())
    for i in range(len(train_path)):
        logger.info("Label %d Alphabet Size: %d" % (i, label_word2index_list[i].size()))
    return word_word2index, char_word2index, label_word2index_list


def read_data(data_path, word_word2index, char_word2index, label_word2index, label_type, label_bucket,
              max_size=None, normalize_digits=True, use_lm=False, use_elmo=False):
    _buckets = label_bucket[label_type]
    max_length = 0
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % data_path)
    counter = 0
    reader = DataReader(data_path, word_word2index, char_word2index, label_word2index, use_elmo)
    inst = reader.get_next(normalize_digits)
    while inst is not None and (not max_size or counter < max_size):
        max_length = max(max_length, inst[6])
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)
        inst_size = len(inst[0])
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                if use_elmo:
                    words = inst[0]
                else:
                    words = inst[1]
                if use_lm:
                    data[bucket_id].append([words, inst[3], inst[5], inst[7], inst[8]])
                else:
                    data[bucket_id].append([words, inst[3], inst[5]])
                max_len = max([len(char_seq) for char_seq in inst[2]])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break
        inst = reader.get_next(normalize_digits)
    reader.close()
    print("Total number of data: %d" % counter)
    print("Max length: %d" % max_length)
    return data, max_char_length


def read_data_to_tensor(data_path, word_word2index, char_word2index, label_word2index, device, label_type, label_bucket,
                        max_size=None, normalize_digits=True, use_lm=False, use_elmo=False):
    data, max_char_length = read_data(data_path, word_word2index, char_word2index, label_word2index, label_type,
                                      label_bucket, max_size=max_size, normalize_digits=normalize_digits, use_lm=use_lm,
                                      use_elmo=use_elmo)
    _buckets = label_bucket[label_type]
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    data_variable = []
    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_variable.append((1, 1))
            continue
        bucket_length = _buckets[bucket_id]
        char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id] + NUM_CHAR_PAD)
        if not use_elmo:
            wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        nid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        if use_lm:
            widfw_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
            widbw_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        lengths = np.empty(bucket_size, dtype=np.int64)
        ws_list = []
        for i, inst in enumerate(data[bucket_id]):
            if use_lm:
                wids, cid_seqs, nids, widfws, widbws = inst
            else:
                wids, cid_seqs, nids = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            if use_elmo:
                ws_list.append(wids)
            else:
                wid_inputs[i, :inst_size] = wids
                wid_inputs[i, inst_size:] = PAD_ID_WORD
            if use_lm:
                widfw_inputs[i, :inst_size] = widfws
                widfw_inputs[i, inst_size:] = PAD_ID_WORD
                widbw_inputs[i, :inst_size] = widbws
                widbw_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # ner ids
            nid_inputs[i, :inst_size] = nids
            nid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
        if use_elmo:
            words = batch_to_ids(ws_list)
        else:
            words = torch.from_numpy(wid_inputs)
        chars = torch.from_numpy(cid_inputs)
        ners = torch.from_numpy(nid_inputs)
        masks = torch.from_numpy(masks)
        lengths = torch.from_numpy(lengths)
        words = words.to(device)
        chars = chars.to(device)
        ners = ners.to(device)
        masks = masks.to(device)
        lengths = lengths.to(device)
        if use_lm:
            wordfws = torch.from_numpy(widfw_inputs)
            wordbws = torch.from_numpy(widbw_inputs)
            wordfws = wordfws.to(device)
            wordbws = wordbws.to(device)
            data_variable.append((words, chars, ners, masks, lengths, wordfws, wordbws))
        else:
            data_variable.append((words, chars, ners, masks, lengths))
    return data_variable, bucket_sizes


def get_batch_variable(data, batch_size, use_lm=False):
    data_variable, bucket_sizes = data
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]
    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
    if use_lm:
        words, chars, ners, masks, lengths, wordfws, wordbws = data_variable[bucket_id]
    else:
        words, chars, ners, masks, lengths = data_variable[bucket_id]
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    index = torch.randperm(bucket_size).long()[:batch_size]
    if words.is_cuda:
        index = index.cuda()
    words = words[index]
    if use_lm:
        return words, chars[index], ners[index], masks[index], lengths[index], wordfws[index], wordbws[index]
    else:
        return words, chars[index], ners[index], masks[index], lengths[index]


def iterate_batch_variable(data, batch_size, label_type, label_bucket, use_lm=False, shuffle=False):
    _buckets = label_bucket[label_type]
    data_variable, bucket_sizes = data
    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle(bucket_indices)
    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue
        if use_lm:
            words, chars, ners, masks, lengths, wordfws, wordbws = data_variable[bucket_id]
        else:
            words, chars, ners, masks, lengths = data_variable[bucket_id]
        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            if words.is_cuda:
                indices = indices.cuda()
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            if use_lm:
                yield words[excerpt], chars[excerpt], ners[excerpt], masks[excerpt], lengths[excerpt], \
                      wordfws[excerpt], wordbws[excerpt]
            else:
                yield words[excerpt], chars[excerpt], ners[excerpt], masks[excerpt], lengths[excerpt]


def evaluate_f1(output_file, out_path, uid):
    score_file = out_path + "/score_%s" % str(uid)
    input = open(output_file)
    output = open(score_file, 'w')
    p = subprocess.Popen(shlex.split("perl conlleval.pl"), stdin=input, stdout=output)
    p.wait()
    input.close()
    output.close()
    with open(score_file, 'r') as f:
        f.readline()
        line = f.readline()
        fields = line.split(";")
        acc = float(fields[0].split(":")[1].strip()[:-1])
        precision = float(fields[1].split(":")[1].strip()[:-1])
        recall = float(fields[2].split(":")[1].strip()[:-1])
        f1 = float(fields[3].split(":")[1].strip())
    return acc, precision, recall, f1


def parse_bool(string):
    if string == 'True':
        string = True
    elif string == 'False':
        string = False
    else:
        raise ValueError('Unknown boolean: %s' % string)
    return string
