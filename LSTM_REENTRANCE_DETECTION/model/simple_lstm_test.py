import torch
import torch.nn as nn
import tqdm
import pandas as pd
import numpy as np
import torch.optim as optim
import load_data

contract, vocab_size, word_embeddings, train_iter, valid_iter, test_iter, train_size, valid_size = load_data.load_dataset()


# 封装迭代器
class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars  # 传入自变量x列表和因变量y列表

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)  # 在这个封装中只有一个自变量

            if self.y_vars is not None:  # 把所有因变量cat成一个向量
                temp = [getattr(batch, feat).unsqueeze(1) for feat in self.y_vars]
                y = torch.cat(temp, dim=1).float()
            else:
                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)


train_dl = BatchWrapper(train_iter, "sequence_text", ["label"])
valid_dl = BatchWrapper(valid_iter, "sequence_text", ["label"])
test_dl = BatchWrapper(test_iter, "sequence_text", None)


class SimpleLSTMBaseline(nn.Module):
    def __init__(self, hidden_dim, emb_dim=300, num_linear=1):
        super().__init__()
        # 词汇量是 len(contract.vocab)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1)
        self.linear_layers = []
        # 中间fc层
        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers = nn.ModuleList(self.linear_layers)
        # 输出层
        self.predictor = nn.Linear(hidden_dim, 6)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]  # 选择最后一个output
        for layer in self.linear_layers:
            feature = layer(feature)
        preds = self.predictor(feature)
        return preds


em_sz = 100
nh = 500
model = SimpleLSTMBaseline(nh, emb_dim=em_sz)

opt = optim.Adam(model.parameters(), lr=1e-2)
loss_func = nn.BCEWithLogitsLoss()

epochs = 2

for epoch in range(1, epochs + 1):
    running_loss = 0.0
    running_corrects = 0
    model.train()  # 训练模式
    for x, y in tqdm.tqdm(train_dl):  # 由于我们的封装，我们可以直接对数据进行迭代
        opt.zero_grad()
        preds = model(x)
        loss = loss_func(y, preds)
        loss.backward()
        opt.step()

        running_loss += loss.data[0] * x.size(0)

    epoch_loss = running_loss / train_size

    # 计算验证数据的误差
    val_loss = 0.0
    model.eval()  # 评估模式
    for x, y in valid_dl:
        preds = model(x)
        loss = loss_func(y, preds)
        val_loss += loss.data[0] * x.size(0)

    val_loss /= valid_size
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))

test_preds = []
for x, y in tqdm.tqdm(test_dl):
    preds = model(x)
    preds = preds.data.numpy()
    # 模型的实际输出是logit，所以再经过一个sigmoid函数
    preds = 1 / (1 + np.exp(-preds))
    test_preds.append(preds)
    test_preds = np.hstack(test_preds)

df = pd.read_csv("dataset/contract_test.csv")
for i, col in enumerate(["label"]):
    df[col] = test_preds[:, i]

df.drop("sequence_text", axis=1).to_csv("dataset/submission.csv", index=False)
