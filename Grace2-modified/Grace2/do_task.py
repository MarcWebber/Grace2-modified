import json
import os.path
import sys
import torch
from Dataset import SumDataset
from tqdm import tqdm
from run import gVar
import numpy as np
import pickle
from scipy.sparse import coo_matrix


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


args = dotdict({
    'NlLen': 5,  # NlLen_map[sys.argv[2]],
    'CodeLen': 9,  # CodeLen_map[sys.argv[2]],
    'SentenceLen': 10,
    'batch_size': 10,
    'embedding_size': 32,
    'WoLen': 15,
    'Vocsize': 100,
    'Nl_Vocsize': 40,
    'max_step': 3,
    'margin': 0.5,
    'poolsize': 50,
    'Code_Vocsize': 100,
    'seed': 0,
    'lr': 1e-4
})


def load_model(dir=os.path.dirname(__file__) + '/data/model/best_model30_19.ckpt'):  # absolute path
    model = torch.load(dir)
    return model


def do_task():
    model = load_model()
    task_set = SumDataset(args, "task")
    # 停止模型训练
    model = model.eval()
    # devBatch是data[i]变成张量的集合
    projects_names = []
    for name in task_set.task_project_names:
        projects_names.append(name)
    task_sort = []
    for k, devBatch in tqdm(enumerate(task_set.Get_Train(len(task_set)))):
        # 保证每一个元素都是张量
        for i in range(len(devBatch)):
            devBatch[i] = gVar(devBatch[i])
        # 在使用pytorch时，并不是所有的操作都需要进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）。
        # 而对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用
        # with torch.no_grad():，强制之后的内容不进行计算图构建
        with torch.no_grad():
            l, pre, _ = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[5],
                              devBatch[6], devBatch[7], devBatch[8])
            # 选出测试集合，就是将node数列进行处理如果是2，设为True，如果为其他则为False，例如[False, False, False, False, True, True, True]
            resmask = torch.eq(devBatch[0], 2)

            s = -pre
            # 将s中等于0的元素替换为1e9
            s = s.masked_fill(resmask == 0, 1e9)
            # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y。

            pred = s.argsort(dim=-1)
            pred = pred.data.cpu().numpy()

            for i in range(len(pred)):
                lst = pred[i].tolist()[:resmask.sum(dim=-1)[i].item()]
                task_sort.append(lst)
        return task_sort


if __name__ == '__main__':
    tar_dir = sys.argv[1]
    dir_M = tar_dir + '/M.json'
    with open(dir_M, "r", encoding="utf-8") as f_M:
        M = json.load(f_M)

    dir_t = tar_dir + '/t.json'
    with open(dir_t, "r", encoding="utf-8") as f_t:
        t = json.load(f_t)
    t = [x + 10e-8 for x in t]  # avoid time being too small

    dir_CW = tar_dir + '/CW.json'
    with open(dir_CW, "r", encoding="utf-8") as f_CW:
        CW = json.load(f_CW)

    dir_num = tar_dir + '/num.json'
    with open(dir_num, "r", encoding="utf-8") as f_num:
        num = json.load(f_num)
    dir1 = os.path.dirname(__file__) + "/task_data/test_task1"
    d1 = dict()
    d1['testNum'] = num[0]
    d1['methodNum'] = num[1]
    d1['codeNum'] = num[2]
    MTmap = []
    for i in range(d1['methodNum']):
        MTmap.append(3)
    for i in range(d1['testNum']):
        MTmap.append(2)
    Cmap = []
    for i in range(d1['codeNum']):
        Cmap.append(1)
    d1['methodAndTestTypeMap'] = MTmap
    d1['codeTypeMap'] = Cmap
    d1['coverage_weight'] = CW
    d1['jaccard_avg'] = np.zeros(d1['testNum'])
    d1['per_coverage_list'] = np.zeros(d1['testNum'])
    '''
    M = np.zeros((d1['testNum'] + d1['methodNum'] + d1['codeNum'], d1['testNum'] + d1['methodNum'] + d1['codeNum']))
    for i in range(2, 8):
        M[1][i] = 1
        M[i][1] = 1
    for i in range(3, 8):
        M[2][i] = 1
        M[i][2] = 1
    '''
    M = coo_matrix(M)
    d1['matrix'] = M
    with open(dir1 + '/data.pkl', 'wb') as f:
        pickle.dump(d1, f)
    with open(dir1 + '/time.pkl', 'wb') as f:
        pickle.dump(t, f)
    w_t = [0, 0]
    d2 = dict()
    d2['t_weight_list'] = w_t
    with open(dir1 + '/weight.pkl', 'wb') as f:
        pickle.dump(d2, f)
    sort = do_task()
    print(sort[0])

    # with open('D:/eclipse workplace/TCP/result.json', 'w', encoding='utf-8') as f_r:
    #     json.dump(sort[0], f_r)
