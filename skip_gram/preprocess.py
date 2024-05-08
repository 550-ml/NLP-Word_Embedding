import os
from typing import Optional
import pickle
import random
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
class ProcessData:
    """
    数据预处理类
    """
    def __init__(self,
                 data_dir: Optional[str]='/data2/wangtuo/workspace/embedding/data/trans',
                 window_size: Optional[int]=2):
        # 这种数据类型的,初始化一般就是确定数据文件夹赋值
        self.window_size = window_size
        self.data_dir = data_dir
    
    def _cal_frequn_build_dict(self, words_list: Optional[list]=None):
        """
        构建字典
        
        """
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        print('开始统计词频')
        # 统计词频的字典
        words_new_list = []
        word_freq_dict = {}
        for word in words_list:
            if word not in stop_words:
                word = stemmer.stem(word)
                words_new_list.append(word)
                if word not in word_freq_dict:
                    word_freq_dict[word] = 0
                word_freq_dict[word] +=1
        word_freq_dict = sorted(word_freq_dict.items(), key = lambda x:x[1], reverse = True)

        # 映射
        word2id = {}
        id2word = {}
        word2freq = {}
        
        for i, (word, freq) in enumerate(word_freq_dict):
            word2id[word] = i
            id2word[i] = word
            word2freq[word] = freq
        
        # word2id
        id_list = []
        for word in words_new_list:
            id_list.append(word2id[word])
        return word2id, id2word, word2freq, id_list
    
    def _subsampling(self,
                     id_list: Optional[list]=None, 
                word2freq: Optional[dict]=None,
                id2word: Optional[dict]=None,
                t: Optional[float]=1e-3,):
        """
        二次采样算法
        """
        # 二次采样算法
        id_freq_list = []
        len_list = len(id_list)
        for i in range(len(id_list)):
            freq = word2freq[id2word[id_list[i]]]
            z = freq/len_list
            p = ((z/t)**0.5 + 1) * t/z
            if random.random() < p:
                id_freq_list.append(id_list[i])
        return id_freq_list

    def build_vacab(self,
                    init_txt_path: Optional[str]='/data2/wangtuo/workspace/embedding/data/training.txt'):
        """构建词典"""
        print('构建字典')
        # 加载数据
        with open(init_txt_path,'r') as f:
            for line in f:
                line = line.strip()
                words = line.split()
        # 构建字典
        word2id, id2word, word2freq, id_list = self._cal_frequn_build_dict(words)
        # 二次采样
        id_freq_list = self._subsampling(id_list, word2freq, id2word)
        # 保存数据
        pickle.dump(word2id, open(os.path.join(self.data_dir, 'word2id.dat'),'wb'))
        pickle.dump(id2word, open(os.path.join(self.data_dir, 'id2word.dat'),'wb'))
        pickle.dump(word2freq, open(os.path.join(self.data_dir, 'word2freq.dat'),'wb'))
        pickle.dump(id_list, open(os.path.join(self.data_dir, 'id_list.dat'),'wb'))
        pickle.dump(id_freq_list, open(os.path.join(self.data_dir, 'id_freq_list.dat'),'wb'))
        print('构建完毕,已保存')

    def build_train_data(self, 
                         savepath: Optional[str]='/data3/wangtuo/Homework/Embedding/data/trans/train_data.dat'):
        """实际训练集的创建"""
        print('构建训练数据集')
        if self.data_dir is None:
            raise ValueError('请先构建字典')
        
        # 加载数据
        id_freq_list = pickle.load(open(os.path.join(self.data_dir, 'id_freq_list.dat'),'rb'))
        # 构建训练数据集
        dataset = []
        center_word_id = 0
        while center_word_id < len(id_freq_list):
            window_range = (max(0, center_word_id - self.window_size), min(len(id_freq_list), center_word_id + self.window_size + 1))
            candidate_words = [id_freq_list[i] for i in range(window_range[0], window_range[1]) if i != center_word_id]

            if len(candidate_words) == 4:
                dataset.append((id_freq_list[center_word_id], candidate_words))
            center_word_id += 1
            if center_word_id % 100 == 0:
                print(f'已经处理了{center_word_id}个单词')

        pickle.dump(dataset, open(savepath, 'wb'))
        print('数据集构建完毕')

if __name__ == '__main__':
    data = ProcessData(data_dir='/data3/wangtuo/Homework/Embedding/data/trans')
    data.build_vacab(init_txt_path='/data3/wangtuo/Homework/Embedding/data/training.txt')
    data.build_train_data()