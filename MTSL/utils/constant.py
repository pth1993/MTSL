import re


MAX_CHAR_LENGTH = 45
NUM_CHAR_PAD = 2
DIGIT_RE = re.compile(r"\d")
PAD_WORD = "_PAD_WORD"
PAD_START = "_PAD_START"
PAD_END = "_PAD_END"
PAD_LABEL = "_PAD_LABEL"
PAD_CHAR = "_PAD_CHAR"
UNK_ID = 0
PAD_ID_WORD = 1
PAD_ID_CHAR = 1
PAD_ID_TAG = 0
_buckets_chunk = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
_buckets_ner = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 130]
_buckets_fgner = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
_buckets_pos = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 300]
label_bucket = {'chunk': _buckets_chunk, 'ner': _buckets_ner, 'pos': _buckets_pos, 'ontonotes': _buckets_pos}
