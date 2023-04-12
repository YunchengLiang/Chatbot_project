import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import TransalationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import spacy 
import random
import math
import time 

seed = 19
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic= True

spacy_de=spacy.load("de")
spacy_en=spacy.load("en")

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)][::-1]

SRC= Field(tokenize=tokenize_de, init_token='<sos>',end_token='<eos>', lower=True)
TRG= Field(tokenize=tokenize_en, init_token='<sos>',end_token='<eos>', lower=True)

train_data,valid_data,test_data= Multi30k.splits(exts=('.de','.en'),fields=(SRC, TRG))
print(f"Number of training examples:{len(train_data.examples)}")
print(f"Number of valid examples:{len(valid_data.examples)}")
print(f"Number of testing examples:{len(test_data.examples)}")
print(vars(train_data.examples[0]))

SRC.build_vocab(train_data,min_freq=2)
TRG.build_vocab(train_data,min_freq=2)
print(f'Unique tokens in source(de) vocabulary: {len(SRC.vocab)}')
print(f'Unique tokens in target(en) vocabulary: {len(TRG.vocab)}')

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE=128
train_iterator, valid_iterator, test_iterator= BucketIterator.splits(train_data,valid_data,test_data
                                                                     batch_size=BATCH_SIZE, device=device)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim,n_layers,dropout):
        super().__init__()
        self.input_dim=input_dim
        self.emb_dim=emb_dim
        self.hid_dim=hid_dim
        self.n_layers=n_layers
        self.dropout= dropout

        self.embedding= nn.Embedding(input_dim,emb_dim)
        self.rnn=nn.LSTM(emb_dim,n_layers,dropout=dropout)
        self.dropout=nn.Dropout(dropout)

    def forward(self, src):
        embedding=self.embedding(src)
        embedded=self.dropout(embedding)
        outputs, (hidden,cell)=self.rnn(embedded)
        return hidden, cell
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim,n_layers,dropout):
        super().__init__()
        self.output_dim=output_dim
        self.emb_dim=emb_dim
        self.hid_dim=hid_dim
        self.n_layers=n_layers
        self.dropout= dropout

        self.embedding= nn.Embedding(output_dim,emb_dim)
        self.rnn=nn.LSTM(emb_dim,n_layers,dropout=dropout)
        self.out= nn.Linear(hid_dim,output_dim)
        self.dropout=nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input=input.unsqueeze(0)
        embedding=self.embedding(input)
        embedded=self.dropout(embedding)
        output, (hidden,cell)=self.rnn(embedded,(hidden,cell))
        prediction= self.out(output.squeeze(0))
        return prediction