import re
import torch
import torch.nn as nn
from model.LSTM import LSTMClassifier
from torch.autograd import Variable

text = """mapping (address => uint) public balances;
function buyTokens()
balances[msg.sender] += msg.value;
TokensBought(msg.sender, msg.value);
if (balances[msg.sender] < _amount)
balances[_to]=_amount;
balances[msg.sender]-=_amount;
TokensTransfered(msg.sender, _to, _amount);
function withdraw(address _recipient) returns (bool)
if (balances[msg.sender] == 0){
InsufficientFunds(balances[msg.sender],balances[msg.sender]);}
PaymentCalled(_recipient, balances[msg.sender]);
if (_recipient.call.value(balances[msg.sender])())
balances[msg.sender] = 0;
function withdraw()
uint transferAmt = 1 ether; 
if (!msg.sender.call.value(transferAmt)()) throw;"""

## clean data
text = re.sub(r"[^A-Za-z0-9^,!;\/'+-=_{}()[]", " ", text)
text = re.sub(r",", " , ", text)
text = re.sub(r"\.", " ", text)
text = re.sub(r"!", " ! ", text)
text = re.sub(r"\/", " ", text)
text = re.sub(r"\^", " ^ ", text)
text = re.sub(r"\+", " + ", text)
text = re.sub(r"\-", " - ", text)
text = re.sub(r"\(", " ( ", text)
text = re.sub(r"\)", " ) ", text)
text = re.sub(r"\[", " [ ", text)
text = re.sub(r"\]", " ] ", text)
text = re.sub(r"\{", " { ", text)
text = re.sub(r"\}", " } ", text)
text = re.sub(r"\=", " = ", text)
text = re.sub(r"'", " ", text)
text = re.sub(r";", " ; ", text)
text = re.sub(r":", " : ", text)

print(text)

text = text.split()
print(text)

vocab = set(text)
print(vocab)
vocab_size = len(vocab)
print(vocab_size)

word_to_ix = {word: i for i, word in enumerate(vocab)}
print(word_to_ix)

embeds = nn.Embedding(39, 100)
print(embeds(Variable(torch.LongTensor(vocab))))

data = []
for i in range(2, len(text) - 2):
    context = [text[i - 2], text[i - 1],
               text[i + 1], text[i + 2]]
    target = text[i]
    data.append((context, target))

print(data[:5])
print(data[0][0])


# prepare data
def make_sequence_vector(seq, word_to_ix):
    idxs = [word_to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


make_sequence_vector(data[0][0], word_to_ix)


# 梯度裁剪（Gradient Clipping）
def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


# train the model
embedding_size = 6
hidden_size = 6
input_size = 28
num_layers = 2
out_size = 2
batch_size = 10
num_epochs = 2
learning_rate = 0.003
model = LSTMClassifier(batch_size, out_size, hidden_size, vocab_size, embedding_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# See what the scores are before training
with torch.no_grad():
    inputs = make_sequence_vector(data[0][0], word_to_ix)

