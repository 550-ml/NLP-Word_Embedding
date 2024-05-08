import pickle
import os
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from model import Word2vec, SkipGramModel
from torch.optim import Adam
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', type=str, default='/data3/wangtuo/Homework/Embedding/data/trans')
    parse.add_argument('--save_path', type=str, default='/data3/wangtuo/Homework/Embedding/model')
    parse.add_argument('--model_name', type=str, default='sgn')
    parse.add_argument('--embedding_size', type=int, default=600)
    parse.add_argument('--negative_word_num', type=int, default=10)
    parse.add_argument('--batch_size', type=int, default=4096)
    parse.add_argument('--epoch', type=int, default=54)
    parse.add_argument('--lr', type=float, default=0.001)
    parse.add_argument('--cuda', type=bool, default=True)
    parse.add_argument('--gpus',type=bool, default=False)
    parse.add_argument('--continue_train', type=bool, default=False)
    return parse.parse_args()

class SGNDataset(Dataset):
    
    def __init__(self, 
                 train_data_path: Optional[str]='/data3/wangtuo/Homework/Embedding/data/trans/train_data.dat'):
        dataset = pickle.load(open(train_data_path,'rb'))
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        in_word, out_word = self.dataset[index]
        return in_word, np.array(out_word)


def train_sgn(args):
    word2id = pickle.load(open(os.path.join(args.data_path, 'word2id.dat'),'rb'))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # 创建模型
    print(len(word2id))
    model = Word2vec(vocab_size=len(word2id), embedding_size=args.embedding_size)
    model_path = os.path.join(args.save_path, '{}.pth'.format(args.model_name))
    skip_gram_model = SkipGramModel(model, len(word2id), args.negative_word_num)
    
    # 加载模型
    if os.path.isfile(model_path) and args.continue_train:
        print('记载模型了')
        skip_gram_model.load_state_dict(torch.load(model_path))
        
    if torch.cuda.device_count() > 1 and args.cuda and args.gpus:
        print(f'使用{torch.cuda.device_count()}个GPU进行训练')
        skip_gram_model = nn.DataParallel(skip_gram_model)
    
    if args.cuda:
        skip_gram_model = skip_gram_model.cuda()
        
    # 优化器
    optim = Adam(skip_gram_model.parameters(), lr=args.lr)
    scheduler = StepLR(optim, step_size=3, gamma=0.1)
    optimpath = os.path.join(args.save_path, '{}.optim.pt'.format(args.model_name))
    if os.path.isfile(optimpath) and args.conti:
        optim.load_state_dict(torch.load(optimpath))
    
    # 开始循环
    for epoch in range(1, args.epoch+1):
        dataset = SGNDataset()
        dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        process_bar = tqdm(dataLoader, desc='Epoch {}'.format(epoch),ncols=0)
        for in_word, out_word in process_bar:
            optim.zero_grad()
            loss = skip_gram_model(in_word, out_word)
            loss.backward()
            optim.step()
            process_bar.set_postfix(loss=loss.item())
        
        scheduler.step()
        
        # 每10个epoch保存一次模型
        if epoch % 3 == 0:
            # 保存模型权重
            idx2vec = model.in_vector.weight.data.cpu().numpy()
            pickle.dump(idx2vec, open(os.path.join(args.save_path, 'idx2vec_{}.dat'.format(epoch)), 'wb'))
            torch.save(skip_gram_model.state_dict(), os.path.join(args.save_path, '{}_epoch_{}.pth'.format(args.model_name, epoch)))
            torch.save(optim.state_dict(), os.path.join(args.save_path, '{}_epoch_{}.pt'.format(args.model_name, epoch)))


if __name__ == '__main__':
    train_sgn(parse_args())