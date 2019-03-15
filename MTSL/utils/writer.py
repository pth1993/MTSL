import codecs


class Writer(object):
    def __init__(self, label_word2index):
        self.__source_file = None
        self.__label_word2index = label_word2index

    def start(self, file_path):
        self.__source_file = codecs.open(file_path, 'w', 'utf-8')

    def close(self):
        self.__source_file.close()

    def write(self, word, predictions, targets, lengths, use_elmo):
        if use_elmo:
            batch_size, _, _ = word.shape
        else:
            batch_size, _ = word.shape
        for i in range(batch_size):
            for j in range(lengths[i]):
                tgt = self.__label_word2index.get_instance(targets[i, j])
                pred = self.__label_word2index.get_instance(predictions[i, j])
                self.__source_file.write('_ %s %s\n' % ( tgt, pred))
            self.__source_file.write('\n')
