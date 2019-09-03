import pandas as pd
import unicodedata, re, string
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# 导入数据

train = pd.read_csv(r'F:\python_pycharm\NLP_Beginner\package\train.tsv', sep='\t')
test = pd.read_csv(r'F:\python_pycharm\NLP_Beginner\package\test.tsv', sep='\t', header=0, index_col=0)
labels = np.array(train['Sentiment'])  # 原本：torch.Tensor( train['Sentiment'])


# %%
##数据处理
def normalize(words):
    def remove_non_ascii(words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def to_lowercase(words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    def remove_punctuation(words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def remove_numbers(words):
        """Remove all interger occurrences in list of tokenized words with textual representation"""
        new_words = []
        for word in words:
            new_word = re.sub("\d+", "", word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_numbers(words)
    return words


# First step - tokenizing phrases
train['Words'] = train['Phrase'].apply(nltk.word_tokenize)  # 句子分词了
# Second step - passing through prep functions
train['Words'] = train['Words'].apply(normalize)

# %%创建词袋，如果单纯创建，还有其他简单方法，CountVectorizer
# First step - tokenizing phrases
word_set = set()
for l in train['Words']:
    for e in l:
        word_set.add(e)  # 这种方法相对于append避免了重复词语的出现
print(len(word_set))
train_bag = {word: ii for ii, word in enumerate(word_set, 1)}

# %%
# 把每个单词转换为其在词袋中对应的位置
train['Tokens'] = train['Words'].apply(lambda l: [train_bag[word] for word in l])
max_len = train['Tokens'].str.len().max()
print(max_len)
# 构建embending输入，单词位置，限制长度 不够补零
all_tokens = np.array([t for t in train['Tokens']])
# Create blank rows
train_index = np.zeros((len(all_tokens), max_len), dtype=int)
# for each phrase, add zeros at the end
for i, row in enumerate(all_tokens):
    train_index[i, :len(row)] = row

# %%
# 划分数据集
split_idx0 = int(len(train) * 0.2)
split_idx1 = int(len(train) * 0.3)
train_x, test_x = train_index[:split_idx0], train_index[split_idx0:split_idx1]
train_y, test_y = labels[:split_idx0], labels[split_idx0:split_idx1]  # 此时还不是tensor
print(len(train_x))
print(len(train_y))
# create Tensor datasets
train_data = TensorDataset(torch.LongTensor(train_x), torch.LongTensor(train_y))  # 构建数据，（数据，标签）
test_data = TensorDataset(torch.LongTensor(test_x), torch.LongTensor(test_y))
# dataloaders
batch_size = 54
# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
print(len(train_loader))
print(len(test_loader))


# %%
# 构建模型
class CNN(nn.Module):  # inheriting from nn.Module!

    def __init__(self):
        super(CNN, self).__init__()

        vocab_size = len(train_bag)
        embedding_dim = 200  # 这样维度不是越来越高了吗？
        n_class = 5  # 类别

        self.embeding = nn.Embedding(vocab_size, embedding_dim)  # （词袋大小，embedding的维度）输出（48*200）48是每个词向量长度
        self.conv1 = nn.Sequential(
            # 输入信号的形式为(N,Cin,H,W) (batch size，in_channelsg个数，H，W分别表示特征图的高和宽）
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 48, 200)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 24, 100)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (32, 24, 100)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (32, 12, 50)
        )
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(32 * 12 * 50, n_class)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.embeding(x).unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x.view(-1, 32 * 12 * 50)
        out = self.out(x)  # 交叉熵损失函数自带sofmax
        return out


model = CNN()  # vocab_size, num_labels
model.cuda()
print(model)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# %%
step = 0
for epoch in range(2):
    for inputs, labels in train_loader:
        step = step + 1
        x = inputs.cuda()
        target = labels.cuda()
        out = model(x)
        loss = loss_function(out, target)  # must be (1. nn output, 2. target), the target label is NOT one-hotted
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 100 == 0:
            pred_y = torch.LongTensor().cuda()
            for inputs, labels in test_loader:
                x = inputs.cuda()
                target = labels.cuda()
                test_output = model(x)
                pred_y_onebatch = torch.max(test_output, 1)[1].view(1, -1)  # .data.cpu()# move the computation in GPU
                pred_y = torch.cat((pred_y, pred_y_onebatch), 1)
            pred_y = pred_y.cpu().numpy()
            # print(np.size(pred_y))
            accuracy = float((pred_y == test_y).astype(int).sum()) / np.size(pred_y)
            print('Epoch: ', epoch, '| step: ', step, '| train loss: %.4f' % loss.data.cpu().numpy(),
                  '| train loss: %.4f', accuracy)
