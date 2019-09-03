import pandas as pd
import unicodedata, re
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# 导入数据d
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

# 划分数据集
split_idx0 = int(len(train) * 0.4)
split_idx1 = int(len(train) * 0.5)
train_x, test_x = train_index[:split_idx0], train_index[split_idx0:split_idx1]
train_y, test_y = labels[:split_idx0], labels[split_idx0:split_idx1]  # 此时还不是tensor
print(len(train_x))
print(len(train_y))
# create Tensor datasets
train_data = TensorDataset(torch.LongTensor(train_x), torch.LongTensor(train_y))  # 构建数据，（数据，标签）
test_data = TensorDataset(torch.LongTensor(test_x), torch.LongTensor(test_y))
# dataloaders
batch_size = 36
# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size,drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size,drop_last=True)
print(len(train_loader))
print(len(test_loader))

# %%
# 构建模型
vocab_size = len(train_bag)+1 #这个加一是真的秀 找了两天还是不明白为什么，但是就是有用
embedding_dim = 300  # 这样维度不是越来越高了吗？
hidden_dim = 256
n_layers = 2
n_classes = 5


class RNN(nn.Module):  # inheriting from nn.Module!
    def __init__(self):
        super(RNN, self).__init__()

        self.embeding = nn.Embedding(vocab_size, embedding_dim)  # （词袋大小，embedding的维度）输出（48*200）48是每个词向量长度

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, input, hidden):
        embeds = self.embeding(input)
        rnn_out, hidden = self.rnn(embeds, hidden)
        rnn_out = rnn_out.transpose(0, 1)  # transpose()调换函数的索引值，可以理解为转置。但是此处代码没有意义啊
        rnn_out = rnn_out[-1]  # 输出必须输出最后 一层
        drip_out=self.dropout(rnn_out)
        out = self.fc(drip_out)
        return out, hidden

    def init_hidden(self):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of RNN
        #hidden = torch.zeros([n_layers, batch_size, hidden_dim]).cuda()
        weight = next(self.parameters()).data


        hidden = weight.new(n_layers, batch_size, hidden_dim).zero_().cuda()

        return hidden


model = RNN()  # vocab_size, num_labels
model.cuda()
print(model)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)



# %%%
model.train()
train_step = 0
train_loss_sum = []
for epoch in range(2):
    hidden = model.init_hidden()
    for inputs, labels in train_loader:
        train_step = train_step + 1
        x = inputs.cuda()
        target = labels.cuda()
        train_out, hidden = model(x, hidden)
        hidden.detach_()
        # must be (1. nn output, 2. target), the target label is NOT one-hotted
        train_loss = loss_function(train_out, target)
        optimizer.zero_grad()  # clear gradients for next train
        train_loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if train_step % 100 == 0:
            train_loss_sum.append(train_loss.item())
            print('Epoch: ', epoch, '| step: ', train_step ,'|train_loss:',train_loss.data.cpu().numpy())

#%%
model.eval()
test_loss_sum = []
num_correct=0
test_step=0
with torch.no_grad():
    #pred_y = torch.LongTensor()#.cuda()
    test_hidden = model.init_hidden()
    for inputs, labels in test_loader:
        test_step=test_step+1
        x = inputs.cuda()
        test_target = labels.cuda()
        test_output, test_hidden = model(x, test_hidden)
        test_loss = loss_function(test_output,test_target)  # must be (1. nn output, 2. target), the target label is NOT one-hotted
        _, pred = torch.max(test_output,1)
        # compare predictions to true label
        correct_tensor = pred.eq(test_target.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)
        if test_step % 25 == 0:
            test_loss_sum.append(test_loss.item())
accuracy = num_correct/len(test_loader.dataset)
print('accuracy:{:.4f}'.format(accuracy))

#%%
import matplotlib.pyplot as plt
# plt.xlim(0, len(test_loss_sum));
# plt.ylim(0, max(test_loss_sum));
plt.plot(test_loss_sum)
plt.plot(train_loss_sum)
plt.title('test_loss');
plt.show()

