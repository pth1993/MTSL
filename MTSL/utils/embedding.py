import numpy as np
import codecs
from constant import DIGIT_RE


def load_embedding_dict(embedding_path, normalize_digits=True):
    """
    load word embeddings from file
    :param embedding_path:
    :return: embedding dict, embedding dimention
    """
    print("loading embedding from %s" % embedding_path)
    embedd_dim = -1
    embedd_dict = dict()
    with codecs.open(embedding_path, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim], dtype=np.float32)
            embedd[:] = tokens[1:]
            word = DIGIT_RE.sub("0", tokens[0]) if normalize_digits else tokens[0]
            embedd_dict[word] = embedd
    return embedd_dict, embedd_dim
