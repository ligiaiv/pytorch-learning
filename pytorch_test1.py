import numpy as np
import torch
import pandas as pd
import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from pprint import pprint
# from keras.preprocessing.text import Tokenizer


# data dir
train_addr = "./database/sentiment-analysis-on-movie-reviews/train.tsv"
# embeddings folder 
test_addr = "./database/sentiment-analysis-on-movie-reviews/test.tsv"

def readfile(filename,filetype,word2idx = {}):
    idx2word = {}
    df = pd.read_csv(train_addr,sep = '\t')
    separated = df['Phrase'].apply(word_tokenize)
    print('oi2')

    print('df phrase shape: ',separated.shape) 

    label = df['Sentiment'].values
    print(label)
    ohk_label =  np.eye(label.max()+1)[label]
    
    print(ohk_label)
    words =   [item for sublist in list(separated) for item in sublist]
    
    max_len = np.max(separated.apply(len).values)
    # for row in separated.values:
    #     map(word2idx.get, row)
    # if filetype is 'test':
    #     return separated,label

    words = list(set(words))
    vocab_size = len(words)
    if filetype == 'train':
        word2idx = {o:i for i,o in enumerate(words)}
        idx2word = {i:o for i,o in enumerate(words)}
    # pprint(idx2word)
    data = np.ndarray((0,max_len))
    # tokenized = [list()]
    print("Processing sentences - sequence to number np.array")
    sentence_index = [list(map(word2idx.get,line)) for line in separated]
    sentence_index = [line + [''] * (max_len - len(line)) for line in sentence_index]
    tokenized=np.array(sentence_index)
    print(tokenized.shape)

    return tokenized,ohk_label,word2idx,idx2word



print('oi')

X_train,y_train,w2idx,idx2w = readfile(train_addr,filetype = 'train')
X_test,y_test,_,_ =readfile(test_addr,filetype = 'test',word2idx = w2idx)
quit()


# This reordering is useful for padding
# def tensor_reorder(data):
#     """reorders tensors from longest to shortest"""
#     lengths = [len(i[0]) for i in data]
#     max_len = max(lengths)
#     lengths = torch.LongTensor(lengths)
#     lengths, perm_idx = lengths.sort(0, descending=True)
#     data = data[perm_idx]
#     return data

# train_data = tensor_reorder(train_data)
# dev_data = tensor_reorder(dev_data)
# test_data = tensor_reorder(test_data)

# MAX_LEN = max(len(train_data[0][0]), len(dev_data[0][0])) # we set the maximum length from the max seq in train and dev 
MAX_LEN  = max(X_train.shape[1],60)
if X_test.shape[1]>MAX_LEN:
    print("reducing test set length")
    X_test = X_test[:,:MAX_LEN]
# print("Data Reordered!")
# print("maximum len of sentences:", MAX_LEN)

class SA_Classifier(nn.Module):
    
    def __init__(self,options):
        super(SA_Classifier,self).__init__()
        self.embedding = nn.Embedding(num_embeddings = options['num_emb'],
                                        embedding_dim = options['emb_dim'],
                                        padding_idx = 0)
        self.lstm = nn.LSTM(input_size = options['emb_dim'],output_size = 100)


