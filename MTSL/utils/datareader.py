import codecs
from constant import MAX_CHAR_LENGTH, NUM_CHAR_PAD, DIGIT_RE, PAD_START, PAD_END


class DataReader(object):
    """
    load data from CoNLL-format file: 1st column is word, 2nd column is label
    """
    def __init__(self, file_path, word_word2index, char_word2index, label_word2index, use_elmo=False):
        self.__source_file = codecs.open(file_path, 'r', 'utf-8')
        self.__word_word2index = word_word2index
        self.__char_word2index = char_word2index
        self.__label_word2index = label_word2index
        self.__use_elmo = use_elmo
        self.__start_id = self.__word_word2index.get_index(PAD_START)
        self.__end_id = self.__word_word2index.get_index(PAD_END)

    def close(self):
        self.__source_file.close()

    def get_next(self, normalize_digits=True):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None
        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            lines.append(line.split('\t'))
            line = self.__source_file.readline()
        length = len(lines)
        if length == 0:
            return None
        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        labels = []
        label_ids = []
        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[0]:
                chars.append(char)
                char_ids.append(self.__char_word2index.get_index(char))
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)
            if self.__use_elmo:
                word = tokens[0]
            else:
                word = DIGIT_RE.sub("0", tokens[0]) if normalize_digits else tokens[0]
            label = tokens[1]
            words.append(word)
            word_ids.append(self.__word_word2index.get_index(word))
            labels.append(label)
            label_ids.append(self.__label_word2index.get_index(label))
        words_fw = word_ids[1:] + [self.__end_id]
        words_bw = [self.__start_id] + word_ids[:-1]
        return words, word_ids, char_seqs, char_id_seqs, labels, label_ids, length, words_fw, words_bw
