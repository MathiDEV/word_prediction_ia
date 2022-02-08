import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys

use_cuda = torch.cuda.is_available()

class WordLSTM(nn.Module):

    def __init__(self, n_hidden=256, n_layers=4, drop_prob=0.3, lr=0.001):
        super().__init__()

        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.emb_layer = nn.Embedding(vocab_len, 200)
        self.lstm = nn.LSTM(200, n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, vocab_len)

    def forward(self, x, hidden):
        embedded = self.emb_layer(x)
        lstm_output, hidden = self.lstm(embedded, hidden)
        # out = self.dropout(lstm_output)
        out = lstm_output.reshape(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if (use_cuda):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden
if use_cuda:
    data = torch.load(sys.argv[1])
else:
    data = torch.load(sys.argv[1], map_location=torch.device('cpu'))
vocab_id = data["dictio"]
get_word = {i: word for i, word in enumerate(vocab_id)}
vocab_len = data["vocab_len"]

def clean_sentence(sentence):
    return '. '.join(map(lambda s: s.strip().capitalize(), sentence.replace(" ,", ",").replace(" .", ".").replace("' ", "'").split('.')))

def predict(model, tkn, h=None):
    x = np.array([[vocab_id[tkn]]])
    inputs = torch.from_numpy(x)
    if use_cuda:
        inputs = inputs.cuda()
    h = tuple([each.data for each in h])
    out, h = model(inputs, h)
    p = F.softmax(out, dim=1).data
    p = p.cpu()
    p = p.numpy()
    p = p.reshape(p.shape[1],)
    top_n_idx = p.argsort()[-2:][::-1]
    sampled_token_index = top_n_idx[random.sample([0, 1], 1)[0]]
    return get_word[sampled_token_index], h


def sample(model, size, prime='françaises , français , mes chers'):
    if use_cuda:
        model.cuda()
    model.eval()
    h = model.init_hidden(1)
    toks = prime.split()
    for t in prime.split():
        token, h = predict(model, t, h)
    toks.append(token)
    for _ in range(size-1):
        token, h = predict(model, toks[-1], h)
        toks.append(token)
    return ' '.join(toks)

model = data["model"]
model.eval()

if use_cuda:
    model.cuda()

print(clean_sentence(sample(model, int(sys.argv[3]), sys.argv[2])))
