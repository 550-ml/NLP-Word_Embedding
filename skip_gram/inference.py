# 推理代码，加载训练好的词向量推理


# 加载模型，返回模型
from typing import Optional
import pickle
import os
import numpy as np
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
def load_dict(dict_folder: Optional[str] = '/data2/wangtuo/workspace/embedding/data/trans'):
    word2id = pickle.load(open(os.path.join(dict_folder,'word2id.dat'),'rb'))
    return word2id
    
    
def load_Word2Vec_model(id2word_file_path: Optional[str]='/data2/wangtuo/workspace/embedding/model/idx2vec_6.dat'):
    word_vector = pickle.load(open(id2word_file_path,'rb'))
    return word_vector
word2id = load_dict(dict_folder='/data3/wangtuo/Homework/Embedding/data/trans')
word_vector = load_Word2Vec_model(id2word_file_path='/data3/wangtuo/Homework/Embedding/model/idx2vec_33.dat')

def get_cos_sim(word1, word2):
    """
    计算余弦相似度
    """
    word1 = stemmer.stem(word1)
    word2 = stemmer.stem(word2)
    
    word1 = word1.lower()
    word2 = word2.lower()
    
    word1_id = word2id.get(word1, None)
    word2_id = word2id.get(word2, None)
    
    if word1_id is None or word2_id is None:
        print("One or both of the words not found in vocabulary.")
        return 0
    word1_vector = word_vector[word1_id]
    word2_vector = word_vector[word2_id]
    
    dot_product = np.dot(word1_vector, word2_vector)
    norm_word1 = np.linalg.norm(word1_vector)
    norm_word2 = np.linalg.norm(word2_vector)
    cosine_similarity = dot_product / (norm_word1 * norm_word2)
    return cosine_similarity

def read_txt(test_txt_path: Optional[str]='/data2/wangtuo/workspace/embedding/data/output.txt',
             output_path:Optional[str]='output7.txt'):
    with open(test_txt_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            words = line.split()
            word1 = words[1]
            word2 = words[2]
            sim = get_cos_sim(word1,word2)
            line_with_sim = line + '\t' + str(sim) + '\n'
            f_out.write(line_with_sim)
            
if __name__ == '__main__':
    read_txt(test_txt_path='/data3/wangtuo/Homework/Embedding/data/output.txt')