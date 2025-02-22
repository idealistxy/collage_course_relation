'''
Description: 
Author: 张轩誉
Date: 2024-06-21 07:59:55
LastEditors: 张轩誉
LastEditTime: 2024-06-21 09:01:44
'''
import re
import pypinyin  # 调⽤pypinyin库对语料进⾏注⾳
import numpy as np
from hmmlearn import hmm


class pinyin(object):
    def __init__(self):
        """
        初始化函数
        """
        self.pi = {}  # 初始状态概率 {word:pro, ..}
        self.A = {}  # 状态转移概率 {phrase:pro, ..}
        self.B = {}  # 观测状态概率 {pinyin:{word:pro, ..}, ..}
        self.dic = {}  # 拼⾳字典 {pinyin: wordwordword}
        self.word2id = {}  # 汉字到id的映射
        self.id2word = {}  # id到汉字的映射
        self.state_index = {}  # 状态（汉字）的索引映射
        self.observation_index = {}  # 观测（拼⾳）的索引映射

    def train_A_B_pi(self):
        """
        求初始状态概率pi、状态转移概率A、观测状态概率B
        :return:
        """
        traindata_path = r"D:/大三课程/机器学习/实验/实验十三/dataset/DataSet/toutiao_cat_data.txt"  # 语料数据集路径
        f = open(traindata_path, encoding='utf-8')
        # 统计单字、词组频数
        sw = {}
        dw = {}
        num_word = 0
        for line in f.readlines():
            line = re.findall('[\u4e00-\u9fa5]+', line)
            for sentence in line:
                pw = ""
                for word in sentence:
                    # 统计汉字到id的映射
                    if word not in self.word2id:
                        self.word2id[word] = num_word
                    # 统计单字
                    if word not in sw:
                        sw[word] = 1
                    else:
                        sw[word] += 1
                    # 统计词组
                    if pw != "":
                        if pw + word not in dw:
                            dw[pw + word] = 1
                        else:
                            dw[pw + word] += 1
                    pw = word
                num_word += 1
        print("Train data loaded.")
        f.close()

        # self.pi # 初始状态概率 {word:pro, ..}
        for word in sw.keys():
            self.pi[word] = sw[word] / num_word

        # self.A # 状态转移概率 {phrase:pro, ..}
        for phrase in dw:
            self.A[phrase] = dw[phrase] / sw[phrase[0]]

            # self.B观测状态概率 {pinyin:{word:pro, ..}, ..}
            # 计算观测状态频数
            for word in sw:
                # lazy_pinyin去除拼⾳平仄
                pinyin = pypinyin.lazy_pinyin(word)[0]
                if pinyin not in self.B.keys():
                    self.B[pinyin] = {word: sw[word]}
                else:
                    self.B[pinyin][word] = sw[word]
                # 计算观测状态频率
            for pinyin in self.B.keys():
                sum_word = sum(self.B[pinyin].values())
                for word in self.B[pinyin]:
                    self.B[pinyin][word] = self.B[pinyin][word] / sum_word

                # self.dic # 拼⾳字典 {pinyin: wordwordword}
                dic_path = r"D:/大三课程/机器学习/实验/实验十三/dataset/DataSet/pinyin2hanzi.txt"  # 拼⾳字典路径
                f = open(dic_path, encoding='utf-8')
                for line in f.readlines():
                    line = re.sub(r'[\ufeff]', '', line).strip().split()
                    self.dic[line[0]] = line[1]
                print(f"Pinyin dictionary (dic): {self.dic}")
                f.close()

                # 翻转拼⾳字典，以便将拼⾳映射回汉字
                pinyin_to_words = {}
                for pinyin, words in self.dic.items():
                    for word in words:
                        if pinyin not in pinyin_to_words:
                            pinyin_to_words[pinyin] = []
                        pinyin_to_words[pinyin].append(word)
                # id到汉字的映射
                self.id2word = {idx: state for idx, state in enumerate(self.word2id.keys())}

    def buildHMM(self):
        """
        求HMM的start_prob、 transmat_prob、emission_prob
        :return: modelpinyin
        """
        # 状态（汉字）、观测（拼⾳）的索引映射
        self.state_index = {state: idx for idx, state in enumerate(self.word2id.keys())}
        n_states = len(self.state_index)
        self.observation_index = {obs: idx for idx, obs in enumerate(self.dic.keys())}
        n_observations = len(self.observation_index)

        # # TODO:求出start_prob、 transmat_prob、emission_prob，构建HMM模型，返回 modelpinyin
        # raise NotImplementedError("此部分需要同学们⾃⾏实现。")

        start_prob = np.zeros(n_states)
        transmat_prob = np.zeros((n_states, n_states))
        emission_prob = np.zeros((n_states, n_observations))

        for word, prob in self.pi.items():
            start_prob[self.state_index[word]] = prob
        print(f"Initial state probabilities (pi)")
        for phrase, prob in self.A.items():
            transmat_prob[self.state_index[phrase[0]], self.state_index[phrase[1]]] = prob
        print(f"State transition probabilities (A)")
        for pinyin, words in self.B.items():
            for word, prob in words.items():
                emission_prob[self.state_index[word], self.observation_index[pinyin]] = prob
        print(f"Observation probabilities (B)")
        # 创建HMM模型
        modelpinyin = hmm.MultinomialHMM(n_components=n_states)
        modelpinyin.startprob_ = start_prob
        modelpinyin.transmat_ = transmat_prob
        modelpinyin.emissionprob_ = emission_prob

        return modelpinyin

    def cal_accuracy(self, text, label):
        """
        计算准确性
        :param text: 预测⽂本
        :param label: 真实⽂本
        :return: 准确性
        """
        num_right = 0
        for i in range(len(text)):
            if text[i] == label[i]:
                num_right += 1
        return num_right / len(text)


if __name__ == '__main__':
    pinyin = pinyin()
    pinyin.train_A_B_pi()
    modelpinyin = pinyin.buildHMM()
    print(modelpinyin.startprob_)
    print(modelpinyin.transmat_)
    print(modelpinyin.emissionprob_)
    # 读取测试⽂件数据
    test_path = r"D:/大三课程/机器学习/实验/实验十三/dataset/DataSet/test1.txt"
    f = open(test_path, encoding='gbk')
    flag = True
    pinyin_list = []
    label_list = []
    for line in f.readlines():
        if flag:
            pinyin_list.append(line.strip('\n').lower())
        else:
            label_list.append(line.strip('\n'))
        flag ^= 1
    # 测试模型
    acc_sum = 0
    for i in range(len(pinyin_list)):
        # 将⼀⾏拼⾳序列转换为列表
        py_list = pinyin_list[i].lower().strip().split()
        # 将拼⾳序列转换为观测序列
        py_observations = np.array([pinyin.observation_index[py] for py in py_list])

        # # TODO:使⽤维特⽐算法预测句⼦，返回 pre_words
        # raise NotImplementedError("此部分需要同学们⾃⾏实现。")

        logprob, state_sequence = modelpinyin.decode(py_observations, algorithm="viterbi")
        pre_words = ''.join([pinyin.id2word[state] for state in state_sequence])

        # 计算准确性
        acc = pinyin.cal_accuracy(pre_words, label_list[i])
        acc_sum += acc
        print(f"pinyin={pinyin_list[i]}")
        print(f"predict={pre_words}")
        print(f"label={label_list[i]}")
        print(f"accuracy={acc}\n")
    # 打印总的准确性
    print(f"average accuracy = {acc_sum / len(pinyin_list)}")
