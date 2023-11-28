import pickle
import time
from torch import optim
from Dataset import SumDataset
import os
import re
from optimizer import ScheduledOptim
from utils import get_apfd
from utils import get_apfdc
import torch.utils.data.dataloader as dl
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from tqdm import tqdm
from Model import *
import numpy as np
import torch
import generate
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

device_ids = list(range(torch.cuda.device_count()))
args = dotdict({
    'NlLen': 5, #NlLen_map[sys.argv[2]],
    'CodeLen': 9, #CodeLen_map[sys.argv[2]],
    'SentenceLen':10,
    'batch_size':10,
    'embedding_size':32,
    'WoLen':15,
    'Vocsize':100,
    'Nl_Vocsize': 40,
    'max_step':3,
    'margin':0.5,
    'poolsize':50,
    'Code_Vocsize':100,
    'seed':0,
    'lr':1e-4
})
os.environ['PYTHONHASHSEED'] = str(args.seed)
train_epoch = 30


# loss_file = open('log/loss-'+str(train_epoch)+'.log', 'a')
# val_file = open('log/val-'+str(train_epoch)+'.log', 'a')

def save_model(model, epoch, dirs= "./data/model"):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    global train_epoch
    findmodel = re.compile(r'best_model{}_[0-9]+.ckpt'.format(train_epoch))
    modelname = ''
    for dir in os.listdir(dirs):
        if re.findall(findmodel, dir):
            modelname = dir
            break
    if modelname:
        os.remove(dirs+'/'+modelname)
    torch.save(model, dirs + '/best_model'+ str(train_epoch)+'_'+str(epoch)+'.ckpt')

def load_model(dirs="./data/model"):
    findmodel= re.compile(r'best_model{}_[0-9]+.ckpt'.format(train_epoch))
    modelname=''
    for dir in os.listdir(dirs):
        if re.findall(findmodel,dir):
            modelname=dir
            break
    assert modelname, 'Weights for saved model not found'
    model = torch.load(dirs+'/'+modelname)
    return model

use_cuda = torch.cuda.is_available()

# 确保data是张量
def gVar(data):
    '''
    将data转换为Tensor类型
    '''
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = gVar(data[i])
        tensor = data
    else:
        assert isinstance(tensor, torch.Tensor)
    if use_cuda:
        tensor = tensor.cuda()
        torch.cuda.empty_cache()


    return tensor

def train():
    '''
    模型训练入口方法
    '''

    # 为CPU中设置种子，生成随机数
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  
    np.random.seed(args.seed + 5)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) 

    # 提升GPU训练速度
    # 设为True就可以大大提升卷积神经网络的运行速度。
    # 但是将会让程序在开始时花费一点额外时间，来为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    torch.backends.cudnn.benchmark = False

    # 将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法。
    # 如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
    torch.backends.cudnn.deterministic = True
    # 训练集合
    train_set = SumDataset(args, "train")
    # 验证集合
    val_set = SumDataset(args, "val")
    # 测试集合
    test_set = SumDataset(args,"test")
    # 单词tokens数
    args.Code_Vocsize = 40 # len(train_set.Code_Voc)
    # 语句token的不一样的数量
    args.Nl_Vocsize = train_set.Nl_Vocsize
    # 测试集合使用的项目id
    # print(dev_set.ids)
    # 构建训练模型
    model = NlEncoder(args)
    if use_cuda:
        print('using GPU')
        model = model.cuda()
        # model = nn.DataParallel(model, device_ids=device_ids,output_device=device_ids[0])
        # model = model.cuda(device_ids[0])

    # 构建优化器
    optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=args.lr), args.embedding_size, 4000)
    
    global train_epoch
    logwritter = open('./log/trainlogs/trainepoch={}.txt'.format(train_epoch),'a')
    # 进行训练
    model = model.train()
    #max_apfd = 0
    max_apfdc = 0
    for epoch in range(train_epoch):
        since = time.time()
        losses = []
        index = 0
        print("----------第{}次训练---------".format(epoch+1))
        logwritter.write("----------第{}次训练---------\n".format(epoch+1))
        for idx, dBatch in enumerate(train_set.Get_Train(args.batch_size)):
            if index == 0:
                model = model.eval()
                val_index = 0
                projects_names = []
                #apfds = []
                apfdcs = []
                val_loss = []
                for k, devBatch in tqdm(enumerate(val_set.Get_Train(args.batch_size))):
                    for i in range(len(devBatch)):
                        devBatch[i] = gVar(devBatch[i])
                    with torch.no_grad():
                        l, pre, _ = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[5], devBatch[6], devBatch[7], devBatch[8])
                        val_l = l.mean()
                        val_loss.append(val_l)
                        resmask = torch.eq(devBatch[0], 2)
                        s = -pre  # -pre[:, :, 1]
                        s = s.masked_fill(resmask == 0, 1e9)
                        pred = s.argsort(dim=-1)
                        pred = pred.data.cpu().numpy()
                        for i in range(len(pred)):
                            project_name = val_set.val_project_names[val_index]
                            projects_names.append(project_name)

                            weight_path = './origin_data/val_data/' + project_name + '/weight.pkl'
                            weight_file = open(weight_path, 'rb')
                            weight = pickle.load(weight_file)
                            mutant_matrx = weight['mutant_matrix']

                            time_path = './origin_data/val_data/' + project_name + '/time.pkl'
                            time_file = open(time_path, 'rb')
                            t = pickle.load(time_file)

                            val_index += 1


                            lst = pred[i].tolist()[:resmask.sum(dim=-1)[i].item()]
                            min_val = 1e9
                            for val in lst:
                                if val < min_val:
                                    min_val = val

                            lst[:] = [x - min_val for x in lst]
                            test_sort = lst
#                            print('test_sort', test_sort)
#                            print('mutant_matrix', mutant_matrx)
                            #_, apfd = get_apfd(test_sort, mutant_matrx)
                            _, apfdc = get_apfdc(test_sort, mutant_matrx, t)
                            #apfds.append(apfd)
                            apfdcs.append(apfdc)
                #print(apfds)
                print(apfdcs)
                #logwritter.write(str(apfds)+'\n')
                logwritter.write(str(apfdcs)+'\n')
                #sum_apfd = 0
                sum_apfdc = 0
                #for a in apfds:
                #    sum_apfd += a
                for a in apfdcs:
                    sum_apfdc += a
                # val_file.write('训练集apfd均值：'+str(sum_apfd/len(apfds)))
                #apfd = sum_apfd / len(apfds)
                apfdc = sum_apfdc / len(apfdcs)
                '''
                if apfd > max_apfd:
                    max_apfd = apfd
                    logwritter.write('在第{}轮训练时获取到了bestmodel'.format(epoch)+'\n')
                    save_model(model,epoch)
                print('训练集apfd均值：', str(sum_apfd/len(apfds)))
                logwritter.write('训练集apfd均值：' + str(sum_apfd / len(apfds))+'\n')
                '''
                if apfdc > max_apfdc:
                    max_apfdc = apfdc
                    logwritter.write('在第{}轮训练时获取到了bestmodel'.format(epoch) + '\n')
                    save_model(model, epoch)
                print('训练集apfdc均值：', str(sum_apfdc / len(apfdcs)))
                logwritter.write('训练集apfdc均值：' + str(sum_apfdc / len(apfdcs)) + '\n')
                val_loss_sum = 0
                for v_l in val_loss:
                    val_loss_sum += v_l
                print('val_loss: ', str(val_loss_sum/len(val_loss)))
                logwritter.write('val_loss: '+ str(val_loss_sum/len(val_loss))+'\n')
            index += 1
            model = model.train()
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            # 输入模型训练，返回损失函数
            loss, pre, _ = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3], dBatch[4], dBatch[5], dBatch[6], dBatch[7], dBatch[8])
            # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0
            optimizer.zero_grad()
            # 损失Tensor取平均值
            loss = loss.mean()
            # 反向传播
            loss.backward()
            losses.append(loss)
            # 更新所有的参数
            optimizer.step_and_update_lr()
            # print('loss', loss)
        loss_sum = 0
        for l in losses:
            loss_sum+=l
        # loss_file.write('epoch_loss'+str(loss_sum/len(losses)))
        print('epoch_loss', loss_sum/len(losses))
        logwritter.write('epoch_loss'+str(loss_sum/len(losses))+'\n')
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logwritter.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)+'\n')
     # 保存模型
#    save_model(model)


def qqqtest():
    # 加载模型
    model = load_model()
    test_set = SumDataset(args, "test")
    # 停止模型训练
    model = model.eval()
    # devBatch是data[i]变成张量的集合
    list_path = './weight_list.txt'
    list_file = open(list_path, 'a')
    projects_names = []
    for name in test_set.test_project_names:
        projects_names.append(name)
        list_file.write(name + '\n')
    global train_epoch
    logwritter = open('./log/testlogs/trainepoch={}.txt'.format(train_epoch),'a')
    for epoch in range(train_epoch):
        since =time.time()
        test_sort = []
        print('第{}次预测开始'.format(epoch+1))
        logwritter.write('第{}次预测开始\n'.format(epoch+1))
        for k, devBatch in tqdm(enumerate(test_set.Get_Train(len(test_set)))):
            # 保证每一个元素都是张量
            for i in range(len(devBatch)):
                devBatch[i] = gVar(devBatch[i])
            # 在使用pytorch时，并不是所有的操作都需要进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）。
            # 而对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用
            # with torch.no_grad():，强制之后的内容不进行计算图构建
            with torch.no_grad():
                l, pre, _ = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[5], devBatch[6], devBatch[7], devBatch[8])
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
                    test_sort.append(lst)
        time_elapsed = time.time() - since
        print('第{}次预测完成，耗时 {:.0f}m {:.0f}s'.format(epoch+1,time_elapsed // 60, time_elapsed % 60))
        logwritter.write('第{}次预测完成，耗时 {:.0f}m {:.0f}s'.format(epoch+1,time_elapsed // 60, time_elapsed % 60)+'\n')
        root_path = './data/test_sequene/' + str(epoch)
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        for i in range(len(projects_names)):
            lst_file = open(root_path + '/' + projects_names[i] + '.pkl', 'wb')
            pickle.dump(test_sort[i], lst_file)



if __name__ == "__main__":

   temp = train()

   qqqtest()