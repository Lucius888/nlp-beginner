import pandas as pd
import unicodedata, re, string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 导入数据
# %%
train = pd.read_csv(r'F:\python_pycharm\NLP_Beginner\task_one\train.tsv', sep='\t')
test = pd.read_csv(r'F:\python_pycharm\NLP_Beginner\task_one\test.tsv', sep='\t', header=0, index_col=0)
labels = np.array(train['Sentiment'])  # 原本：torch.Tensor( train['Sentiment'])


# %%
# 数据归一化处理
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
Vectorizer = CountVectorizer(max_df=0.95, min_df=5,stop_words='english')#去除停用词效果确实好了一点点
# (a,b),（ 单词所在得句子，单词所在词袋中得位置）出现的次数
train_CountVectorizer = Vectorizer.fit_transform(train['Phrase'])
train_bag = Vectorizer.vocabulary_
train_one_hot = train_CountVectorizer.toarray()#纤细模型one-hot,但是RNN/CNN都是索引位置
print(len(train_bag))


# %%
# 划分数据集
split_idx0 = int(len(train) * 0.2)
split_idx1 = int(len(train) * 0.3)
train_x, test_x = train_one_hot[:split_idx0], train_one_hot[split_idx0:split_idx1]
train_y, test_y = labels[:split_idx0], labels[split_idx0:split_idx1]  # 此时还不是tensor
print(len(train_x))
print(len(train_y))
# create Tensor datasets
train_data = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))  # 构建数据，（数据，标签）
test_data = TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y))
# dataloaders
batch_size = 64
# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
print(len(train_loader))
print(len(test_loader))


# %%
# 构建模型
class Linear_Classfy(nn.Module):  # inheriting from nn.Module!

    def __init__(self):
        super(Linear_Classfy, self).__init__()
        self.linear0 = nn.Linear(len(train_bag), 256)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.linear0(x)
        x = self.linear1(x)
        out = self.linear2(x)
        # out = F.softmax(x)#交叉熵损失函数自带sofmax
        return out


model = Linear_Classfy()  # vocab_size, num_labels
model.cuda()
print(model)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.2)

# %%
for epoch in range(1):
    for inputs, labels in train_loader:
        x = inputs.cuda()
        target = labels.cuda()
        out = model(x)
        loss = loss_function(out, target)  # must be (1. nn output, 2. target), the target label is NOT one-hotted
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
    # for i in range(len(train_x)):
    #     x=torch.Tensor(train_x[i])
    #     x=torch.FloatTensor(x).view(1,-1).cuda()#交叉熵imput和target的类型必须分别为FloatTensor，FloatTensor，对应32bit和64bit
    #     #x=torch.unsqueeze(x,0)
    #     target=train_y[i]
    #     target=torch.unsqueeze(target,0).long().cuda()
    #     #target=torch.LongTensor(target)
    #     out = model(x)  # input x and predict based on x
    #     loss = loss_function(out, target)  # must be (1. nn output, 2. target), the target label is NOT one-hotted
    #     optimizer.zero_grad()  # clear gradients for next train
    #     loss.backward()  # backpropagation, compute gradients
    #     optimizer.step()  # apply gradients

# %%
# torch.save(model, 'model_batch_train.pkl')


# %%
# model1 = torch.load('model_batch_train.pkl')
pred_y = torch.LongTensor().cuda()
for inputs, labels in test_loader:
    x = inputs.cuda()
    target = labels.cuda()
    test_output = model(x)
    pred_y_onebatch = torch.max(test_output, 1)[1].view(1, -1)  # .data.cpu()# move the computation in GPU
    pred_y = torch.cat((pred_y, pred_y_onebatch), 1)

# %%
pred_y = pred_y.cpu().numpy()
print(np.size(pred_y))
accuracy = float((pred_y == test_y).astype(int).sum()) / np.size(pred_y)
print('accuracy: ', accuracy)
#     if pred_y==test_y[i].long():
#          true_num=true_num+1
# print('accuracy: ' ,true_num/len(test_x))
# true_num=0
# for i in range(len(test_x)):
#     x_for_test=torch.Tensor(test_x[i])
#     x_for_test=torch.FloatTensor(x_for_test).view(1,-1).cuda()
#     test_y[i].cuda()
#     test_output = model1(x_for_test)
#     pred_y = torch.max(test_output, 1)[1].data.cpu()# move the computation in GPU
#     if pred_y==test_y[i].long():
#          true_num=true_num+1
# print('accuracy: ' ,true_num/len(test_x))
