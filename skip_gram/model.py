# 创建模型
import torch
import torch.nn as nn
from torch import LongTensor
from torch import FloatTensor
import torch.nn.init as init
class Word2vec(nn.Module):
    """
    Word2vec表示词向量的学习
    """
    def __init__(self, vocab_size, embedding_size):
        super(Word2vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.in_vector = nn.Embedding(vocab_size, embedding_size)
        self.out_vector = nn.Embedding(vocab_size, embedding_size)
        # 初始化权重
        self._init_emb()
        self.in_vector.weight.requires_grad = True
        self.out_vector.weight.requires_grad = True

    def _init_emb(self):
        init.xavier_uniform_(self.in_vector.weight.data, gain=1.0)
        init.xavier_uniform_(self.out_vector.weight.data, gain=1.0)
    
    def forward(self, data):
        return self.forward_input(data)
    
    def forward_input(self, data):
        in_vertor = data
        in_vertor = in_vertor.cuda() if self.in_vector.weight.is_cuda else in_vertor
        return self.in_vector(in_vertor)
    
    def forward_output(self, data):
        out_vector = data
        out_vector = out_vector.cuda() if self.out_vector.weight.is_cuda else out_vector
        return self.out_vector(out_vector)

class SkipGramModel(nn.Module):
    def __init__(self, embedding, vocab_size, negative_num=10, weights=None):
        super(SkipGramModel, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.negative_num = negative_num

    def forward(self, inword, outword):
        batch_size = inword.size(0)
        context_size = outword.size(1)
        # 生成负采样的索引
        negword = torch.FloatTensor(batch_size, context_size*self.negative_num).uniform_(0, self.vocab_size-1).long()
        
        # 前向传播
        # inword-> [batch,] outword->[batch, context_size] negword->[batch, context_size*negative_num]
        # in_vectors->[batch, embedding_size] out_vectors->[batch, context_size, embedding_size] neg_vectors->[batch, context_size*negative_num, embedding_size]
        in_vectors = self.embedding.forward_input(inword).unsqueeze(2) # [batch, embedding_size, 1]
        neg_vectors = self.embedding.forward_output(negword).neg()  # [batch, context_size*negative_num, embedding_size]
        out_vectors = self.embedding.forward_output(outword)  # [batch, context_size, embedding_size]
        # 计算损失
        outword_loss = torch.bmm(out_vectors, in_vectors).squeeze().sigmoid().log().mean(1) # [batch, context_size]->[batch]
        negword_loss = torch.bmm(neg_vectors, in_vectors).squeeze().sigmoid().log().view(-1, context_size, self.negative_num).sum(2).mean(1)
        loss = -(outword_loss + negword_loss).mean()
        # print(loss.dtype)
        # print(loss.shape)
        return loss