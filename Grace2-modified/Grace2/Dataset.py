import sys
import torch
import torch.utils.data.dataset as data
import generate
import pickle
import os
'''from nltk import word_tokenize'''
import scipy.sparse as sp


import numpy as np
import re
from tqdm import tqdm
from scipy import sparse
import math
import json


class SumDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, config, dataName, proj="Math", testid=0, lst=[]):

        self.proj = proj
        self.SentenceLen = config.SentenceLen
        self.Nl_Voc = {
            "pad": 0,
            "UnKnow": 1,
            "Test": 2,
            "MethodDeclaration": 3,
            "InterfaceDeclaration": 4,
            "ConstructorDeclaration": 5,
            "VariableDeclaration": 6,
            "LocalVariableDeclaration": 7,
            "FormalParameter": 8,
            "IfStatement": 9,
            "WhileStatement": 10,
            "DoStatement": 11,
            "ForStatement": 12,
            "AssertStatement": 13,
            "BreakStatement": 14,
            "ContinueStatement": 15,
            "ReturnStatement": 16,
            "ThrowStatement": 17,
            "SynchronizedStatement": 18,
            "TryStatement": 19,
            "SwitchStatement": 20,
            "BlockStatement": 21,
            "StatementExpression": 22,
            "TryResource": 23,
            "CatchClause": 24,
            "CatchClauseParameter": 25,
            "SwitchStatementCase": 26,
            "ForControl": 27,
            "EnhancedForControl": 28,
            "Expression": 29,
            "Assignment": 30,
            "TernaryExpression": 31,
            "BinaryOperation": 32,
            'MethodInvocation': 33,
            'Statement': 34,
            'Literal': 35,
            'ClassDeclaration': 36
        }
        self.Nl_Len = config.NlLen
        self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.batch_size = 4  # config.batch_size
        self.data = []
        self.PAD_token = 0
        self.mutant_apfd = []
        self.Nl_Vocsize = 40
        self.test_len = 0
        self.method_len = 0
        self.line_len = 0
        self.dataName = dataName
        self.project_names = []
        self.val_project_names = []
        self.test_project_names = []
        self.train_project_names = []
        self.task_project_names = []
        if dataName == 'train':
            self.data = self.get_data_list('train')
        elif dataName == 'val':
            self.data = self.get_data_list('val')
        elif dataName == 'test':
            self.data = self.get_data_list('test')
        elif dataName == 'task':
            self.data = self.get_data_list('task')
        else:
            pass

    def get_data_list(self, type):
        '''
        新数据获取方法，将数据加入到模型，主要是给七个参数赋值input_node, inputtype, inputad, res, inputtext, linenode, linetype
        :param type: 数据类型trian、test、val
        :return:
        '''
        data = []
        project_name_list = []
        # 初始化需要的8个参数，9（with time）
        allProjectMethodNode = []
        allProjectTestNode = []
        allProjectLineNode = []
        #
        allProjectTestTime = []
        allProjectTestCoverageWeight = []
        allProjectTestJaccard = []
        allProjectPerCoverage = []
        allProjectTestWeight = []
        allProjectMatrix = []

        if type == 'val':
            val_project_names = []
            dir_path = "./origin_data/val_data"
            for project_name in os.listdir(dir_path):
                val_project_names.append(project_name)
                project_name_list.append(project_name)
                file = open(dir_path + "/" + project_name + "/data.pkl", 'rb')
                methodAndTestAndCodeInformation = pickle.load(file, encoding="utf-8")
                self.test_len = max(self.test_len, int(methodAndTestAndCodeInformation['testNum']))
                self.method_len = max(self.method_len, int(methodAndTestAndCodeInformation['methodNum']))
                self.line_len = max(self.line_len, methodAndTestAndCodeInformation['codeNum'])
            project_name_list = val_project_names
            self.val_project_names=val_project_names
        elif type == 'test':
            test_project_names = []
            dir_path = "./origin_data/test_data"
            for project_name in os.listdir(dir_path):
                test_project_names.append(project_name)
                project_name_list.append(project_name)
                file = open(dir_path + "/" + project_name + "/data.pkl", 'rb')
                methodAndTestAndCodeInformation = pickle.load(file, encoding="utf-8")
                self.test_len = max(self.test_len, int(methodAndTestAndCodeInformation['testNum']))
                self.method_len = max(self.method_len, int(methodAndTestAndCodeInformation['methodNum']))
                self.line_len = max(self.line_len, methodAndTestAndCodeInformation['codeNum'])
            project_name_list = test_project_names
            self.test_project_names=test_project_names
        elif type == 'train':
            train_project_names = []
            dir_path = "./origin_data/train_data"
            for project_name in os.listdir(dir_path):
                train_project_names.append(project_name)
                project_name_list.append(project_name)
                file = open(dir_path + "/" + project_name + "/data.pkl", 'rb')
                methodAndTestAndCodeInformation = pickle.load(file, encoding="utf-8")
                self.test_len = max(self.test_len, int(methodAndTestAndCodeInformation['testNum']))
                self.method_len = max(self.method_len, int(methodAndTestAndCodeInformation['methodNum']))
                self.line_len = max(self.line_len, methodAndTestAndCodeInformation['codeNum'])
            project_name_list = train_project_names
            self.train_project_names=train_project_names
        elif type == 'task':
            task_project_names = []
            # TODO: 修改路径
            dir_path = "../../Grace2-modified/Grace2/task_data" # absolute path
            dir_path = os.path.dirname(__file__) + '/task_data'
            for project_name in os.listdir(dir_path):
                task_project_names.append(project_name)
                project_name_list.append(project_name)
                file = open(dir_path + "/" + project_name + "/data.pkl", 'rb')
                methodAndTestAndCodeInformation = pickle.load(file, encoding="utf-8")
                self.test_len = max(self.test_len, int(methodAndTestAndCodeInformation['testNum']))
                self.method_len = max(self.method_len, int(methodAndTestAndCodeInformation['methodNum']))
                self.line_len = max(self.line_len, methodAndTestAndCodeInformation['codeNum'])
            project_name_list = task_project_names
            self.task_project_names = task_project_names

        for project_name in project_name_list:
            self.project_names.append(project_name)
            file = open(dir_path + "/" + project_name + "/data.pkl", 'rb')
            methodAndTestAndCodeInformation = pickle.load(file, encoding="utf-8")
            file = open(dir_path + "/" + project_name + "/weight.pkl", 'rb')
            weight = pickle.load(file, encoding="utf-8")
            #
            file = open(dir_path + "/" + project_name + "/time.pkl", 'rb')
            time = pickle.load(file, encoding="utf-8")

            methodNum = int(methodAndTestAndCodeInformation['methodNum'])
            codeNum = int(methodAndTestAndCodeInformation['codeNum'])
            testNum = int(methodAndTestAndCodeInformation['testNum'])
            methodAndTestNum = methodNum + testNum
            methodAndCodeNum = methodNum + codeNum
            methodAndTestTypeMap = methodAndTestAndCodeInformation['methodAndTestTypeMap']
            oneProjectMethodNode = []
            for i in range(methodNum):
                oneProjectMethodNode.append(methodAndTestTypeMap[i])
            allProjectMethodNode.append(oneProjectMethodNode)

            oneProjectTestNode = []
            for i in range(methodNum, methodAndTestNum):
                oneProjectTestNode.append(methodAndTestTypeMap[i])
            allProjectTestNode.append(oneProjectTestNode)

            allProjectLineNode.append(methodAndTestAndCodeInformation['codeTypeMap'])

            allProjectTestTime.append(time)
            allProjectTestCoverageWeight.append(methodAndTestAndCodeInformation['coverage_weight'])

            allProjectTestJaccard.append(methodAndTestAndCodeInformation['jaccard_avg'])

            allProjectPerCoverage.append(methodAndTestAndCodeInformation['per_coverage_list'])

            allProjectTestWeight.append(weight['t_weight_list'])

            allProjectMatrix.append(methodAndTestAndCodeInformation['matrix'])
        for i in range(len(allProjectTestNode)):
            allProjectTestNode[i] = self.pad_seq(allProjectTestNode[i], self.test_len)
            # allProjectTestNode[i] = torch.from_numpy(np.array(allProjectTestNode[i]))
        data.append(allProjectTestNode)

        for i in range(len(allProjectMethodNode)):
            allProjectMethodNode[i] = self.pad_seq(allProjectMethodNode[i], self.method_len)
            # allProjectMethodNode[i] = torch.from_numpy(np.array(allProjectMethodNode[i]))
        data.append(allProjectMethodNode)

        for i in range(len(allProjectLineNode)):
            allProjectLineNode[i] = self.pad_seq(allProjectLineNode[i], self.line_len)
            # allProjectLineNode[i] = torch.from_numpy(np.array(allProjectLineNode[i]))
        data.append(allProjectLineNode)

        #
        for i in range(len(allProjectTestTime)):
            allProjectTestTime[i] = self.pad_seq(allProjectTestTime[i], self.test_len)
            # allProjectTestTime[i] = torch.from_numpy(np.array(allProjectTestTime[i]))
        data.append(allProjectTestTime)

        for i in range(len(allProjectTestCoverageWeight)):
            allProjectTestCoverageWeight[i] = self.pad_seq(allProjectTestCoverageWeight[i], self.test_len)
            # allProjectTestCoverageWeight[i] = torch.from_numpy(np.array(allProjectTestCoverageWeight[i]))
        data.append(allProjectTestCoverageWeight)

        for i in range(len(allProjectTestJaccard)):
            allProjectTestJaccard[i] = self.pad_seq(allProjectTestJaccard[i], self.test_len)
            # allProjectTestJaccard[i] = torch.from_numpy(np.array(allProjectTestJaccard[i]))
        data.append(allProjectTestJaccard)

        for i in range(len(allProjectPerCoverage)):
            allProjectPerCoverage[i] = self.pad_seq(allProjectPerCoverage[i], self.test_len)
            # allProjectPerCoverage[i] = torch.from_numpy(np.array(allProjectPerCoverage[i]))
        data.append(allProjectPerCoverage)

        for i in range(len(allProjectTestWeight)):
            allProjectTestWeight[i] = self.pad_seq(allProjectTestWeight[i], self.test_len)
            # allProjectTestWeight[i] = torch.from_numpy(np.array(allProjectTestWeight[i]))
        data.append(allProjectTestWeight)
        # for i in range(len(allProjectMatrix)):
        #     allProjectMatrix[i] = torch.from_numpy(allProjectMatrix[i].todense().A)
        data.append(allProjectMatrix)
        return data

    def pad_seq(self, seq, maxlen):
        '''
        让所有序列变成seq集合maxlen长度，后边补0
        '''
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq

    def __getitem__(self, offset):
        ans = []
        for i in range(len(self.data)):
            ans.append(np.array(self.data[i][offset]))
        return ans

    def __len__(self):
        return len(self.data[0])

    # 把数组变成张量
    def Get_Train(self, batch_size):  # bitch_size每次取数据的多少
        data = self.data
        loaddata = data
        # 记录总共取的次数
        batch_nums = int(len(data[0]) / batch_size)
        if True:
            if self.dataName == 'train':
                # 假如一共有10个项目，则shuffle为0-9随机组成的序列，如[1, 2, 6, 7, 8, 0, 9, 4, 3, 5]
                shuffle = np.random.permutation(range(len(loaddata[0])))
            else:
                # 函数返回一个有终点和起点的固定步长的排列，如arange(3)-> [0,1,2]
                # arange(1,3) -> [1,2]
                # arange(1,3,0.5) -> [1, 1.5, 2]
                shuffle = np.arange(len(loaddata[0]))
            # 分批次获取数据
            for i in range(batch_nums):
                ans = []
                # 把数组编程张量
                for j in range(len(data)):
                    if j != 8:
                        # 就是随机获取data中的数据，但是因为shuffle是已经随机固定的，所以data获取时，是可以对应的
                        tmpd = np.array(data[j])[shuffle[batch_size * i: batch_size * (i + 1)]]
                        # 方法把数组转换成张量
                        ans.append(torch.from_numpy(np.array(tmpd)))
                    else:
                        # 处理三维矩阵
                        ids = []
                        v = []
                        for idx in range(batch_size * i, batch_size * (i + 1)):
                            for p in range(len(data[j][shuffle[idx]].row)):
                                # ids存放的数据类似于[0, 2, 4]
                                ids.append(
                                    [idx - batch_size * i, data[j][shuffle[idx]].row[p], data[j][shuffle[idx]].col[p]])
                                # v存放val的值
                                v.append(data[j][shuffle[idx]].data[p])
                        # 类型转换, 将list ,numpy转化为tensor
                        ans.append(torch.sparse.FloatTensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size([batch_size, self.test_len + self.method_len + self.line_len,self.test_len + self.method_len + self.line_len])))
                yield ans
            # 如果到了最后一批数据
            if batch_nums * batch_size < len(data[0]):
                ans = []
                for j in range(len(data)):
                    if j != 8:
                        tmpd = np.array(data[j])[shuffle[batch_nums * batch_size:]]
                        ans.append(torch.from_numpy(np.array(tmpd)))
                    else:
                        ids = []
                        v = []
                        for idx in range(batch_size * batch_nums, len(data[0])):

                            for p in range(len(data[j][shuffle[idx]].row)):
                                ids.append([idx - batch_size * batch_nums, data[j][shuffle[idx]].row[p],
                                            data[j][shuffle[idx]].col[p]])
                                v.append(data[j][shuffle[idx]].data[p])
                        ans.append(torch.sparse.FloatTensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size([len(data[0]) - batch_size * batch_nums, self.test_len + self.method_len + self.line_len,self.test_len + self.method_len + self.line_len])))
                yield ans
