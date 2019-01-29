import pandas as pd
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim

pd.set_option('display.width', 1000)
# GPU
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the data
train_data_ex = pd.read_csv("data/train.csv").head(2)
valid_data_ex = pd.read_csv("data/valid.csv").head(2)
test_data_ex = pd.read_csv("data/test.csv").head(2)

# Declaring Fields
tokenize = lambda x: x.split()
TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = Field(sequential=False, use_vocab=False)

# Creating the Dataset
tv_datafields = [
    ("id", None),
    ("comment_text", TEXT), ("toxic", LABEL),
    ("severe_toxic", LABEL), ("threat", LABEL),
    ("obscene", LABEL), ("insult", LABEL),
    ("identity_hate", LABEL)
]

train_data, valid_data = TabularDataset.splits(
    path="data",  # the root directory where the data lies
    train="train.csv", validation="valid.csv",
    format='csv',
    skip_header=True,
    fields=tv_datafields
)

tst_datafields = [
    ("id", None),
    ("comment_text", TEXT)
]

test_data = TabularDataset(
    path="data/test.csv",
    format='csv',
    skip_header=True,
    fields=tst_datafields
)

TEXT.build_vocab(train_data)
TEXT.vocab.freqs.most_common(10)
print(train_data[0])
print(train_data[0].__dict__.keys)
print(train_data[0].comment_text[:3])

# Creating the Iterator
train_iter, val_iter = BucketIterator.splits(
    (train_data, valid_data),
    batch_size=64,
    device=device,
    sort_key=lambda x: len(x.comment_text),
    sort_within_batch=False,
    repeat=False
)

batch = next(train_iter.__iter__());batch

batch.__dict__.keys()

test_iter = Iterator(test_data, batch_size=64, device=device, sort=False, sort_within_batch=False, repeat=False)


# Wrapping the Iterator
class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)

            if self.y_vars is not None:
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)


train_dl = BatchWrapper(train_iter, "comment_text",
                        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
valid_dl = BatchWrapper(val_iter, "comment_text",
                        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
test_dl = BatchWrapper(test_iter, "comment_text", None)

next(train_dl.__iter__())


# train a text classifier
class SimpleBiLSTMBaseline(nn.Module):
    def __init__(self, hidden_dim, emb_dim=300,
                 spatial_dropout=0.05, recurrent_dropout=0.1, num_linear=1):
        super().__init__()  # don't forget to call this!
        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1, dropout=recurrent_dropout)
        self.linear_layers = []
        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim, 6)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
        preds = self.predictor(feature)
        return preds


em_sz = 100
hidden_number = 500
layer_number = 3
model = SimpleBiLSTMBaseline(hidden_number, emb_dim=em_sz)
model.cuda()

# The training loop
opt = optim.Adam(model.parameters(), lr=1e-2)
loss_func = nn.BCEWithLogitsLoss()  # Multi-label loss
epochs = 4
for epoch in range(1, epochs + 1):
    running_loss = 0.0
    running_corrects = 0
    model.train()  # turn on training mode
    for x, y in tqdm.tqdm(train_dl):  # thanks to our wrapper, we can intuitively iterate over our data!
        opt.zero_grad()

        preds = model(x)
        loss = loss_func(preds, y)
        loss.backward()
        opt.step()

        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(train_data)

    # calculate the validation loss for this epoch
    val_loss = 0.0
    model.eval()  # turn on evaluation mode
    for x, y in valid_dl:
        preds = model(x)
        loss = loss_func(preds, y)
        val_loss += loss.item() * x.size(0)

    val_loss /= len(valid_data)
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
