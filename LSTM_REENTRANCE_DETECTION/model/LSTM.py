import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# GPU
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, out_size, hidden_size, vocab_size, embedding_size, num_layers):
        super(LSTMClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_layers = num_layers

        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)

        # LSTM (GRU)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        # FC
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, sentence, batch_size=None):
        embeds = self.word_embeddings(sentence)
        embeds = embeds.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device))
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device))
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))
        output, (h_n, c_n) = self(embeds, (h_0, c_0))
        print(output.size(), h_n.size(), c_n.size())
        final_output = self.fc(h_n[-1])
        final_score = F.log_softmax(final_output, dim=1)
        return final_output, final_score
