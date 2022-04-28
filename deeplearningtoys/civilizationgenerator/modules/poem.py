'''
Function:
    定义诗歌生成模型
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import torch.nn as nn


'''诗歌生成模型'''
class Poem(nn.Module):
    def __init__(self, vocabulary_dim, embedding_dim=256, hidden_dim=512, num_layers=3):
        super(Poem, self).__init__()
        self.vocabulary_dim = vocabulary_dim 
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # 定义一些层
        self.embedding = nn.Embedding(vocabulary_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim, vocabulary_dim)
    '''forward'''
    def forward(self, inputs, hidden=None):
        seq_len, batch_size = inputs.size()
        if hidden is None:
            hidden = inputs.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float(), inputs.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        h_0, c_0 = hidden
        embeds = self.embedding(inputs)
        outputs, hidden = self.lstm(embeds, (h_0, c_0))
        outputs = self.linear(outputs.view(seq_len * batch_size, -1))
        return outputs, hidden