import  os
import  pickle
import numpy as np
import pandas


def build_char_matrix(char2idx, char_dim):
    embedding_matrix_file_name = '{0}_dependency_matrix.pkl'.format(str(char_dim))
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading char vectors ...')
        embedding_matrix = np.zeros((len(char2idx), char_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(char_dim), 1 / np.sqrt(char_dim), (1, char_dim))
    # embedding_matrix[1, :] = np.random.uniform(-1, 0.25, (1, dependency_dim))
        print('building char_matrix:', embedding_matrix_file_name)
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def fit_on_text(self, text):
        for word in text:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in text]
        if len(sequence) == 0:
            sequence = [0]
        return sequence

class DatesetReader:
    def __init__(self,train_data):
        # print(train_data)
        all_char = []
        for index, row in train_data.iterrows():
            # print(row["SMILES"])
            char_list = list(row["SMILES"])
            # print(char_list)
            all_char+= char_list
        self.char_Tokenizer = Tokenizer()
        self.char_Tokenizer.fit_on_text(all_char)
        self.char_matrix = build_char_matrix(self.char_Tokenizer.word2idx, 50)

    def __read_data__(self,text):
        all_data = []
        # print("11111111111111")
        for index, row in text.iterrows():
            char_list = list(row["SMILES"])
            char_indices = self.char_Tokenizer.text_to_sequence(char_list)
            other_feature = list(row[1:21])

            tags = row[-1]
            data = {
            "char_indices" : char_indices,
            "other_feature":other_feature,
            "tags":tags
            }
            all_data.append(data)
        return all_data



















