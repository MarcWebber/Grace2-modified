import torch.nn as nn
import torch.nn.functional as F
import torch

from transformer import TransformerBlock, rightTransformerBlock
from basicLayers import PositionalEmbedding, LayerNorm


class NlEncoder(nn.Module):
    def __init__(self, args):
        super(NlEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        # nn.ModuleList，它是一个储存不同 module，并自动将每个 module 的
        # parameters 添加到网络之中的容器。你可以把任意 nn.Module 的子类
        # (比如 nn.Conv2d, nn.Linear 之类的) 加到这个 list 里面，
        # 方法和 Python 自带的 list 一样，无非是 extend，append 等操作。
        # 但不同于一般的 list，加入到 nn.ModuleList 里面的 module
        # 是会自动注册到整个网络上的，同时 module 的 parameters
        # 也会自动添加到整个网络中。
        self.transformerBlocks = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])
        # 字典中共有args.Nl_Vocsize个词，用self.embedding_size-1维度向量表示
        self.token_embedding = nn.Embedding(args.Nl_Vocsize, self.embedding_size)
        self.token_embedding1 = nn.Embedding(args.Nl_Vocsize, self.embedding_size - 1)
        self.token_embedding2 = nn.Embedding(args.Nl_Vocsize, self.embedding_size - 2)
        self.token_embedding3 = nn.Embedding(args.Nl_Vocsize, self.embedding_size - 3)
        self.token_embedding4 = nn.Embedding(args.Nl_Vocsize, self.embedding_size - 4)

        self.text_embedding = nn.Embedding(20, self.embedding_size)
        self.transformerBlocksTree = nn.ModuleList(
            [rightTransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])
        self.resLinear = nn.Linear(self.embedding_size, 2)
        self.pos = PositionalEmbedding(self.embedding_size)
        # 交叉熵损失函数
        self.loss = nn.CrossEntropyLoss()
        self.norm = LayerNorm(self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size // 2, int(self.embedding_size / 4), batch_first=True,
                            bidirectional=True)
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 10))
        # nn.Linear(in_features, out_features)in_features由输入张量的形状决定，out_features则决定了输出张量的形状
        self.resLinear2 = nn.Linear(self.embedding_size, 1)

    def forward(self, test_node, method_node, line_node, test_time, test_coverage_weight, test_jaccard, test_per_coverage, test_weight, matrix):
        # print('time:', test_time)
        matrix = matrix.float()

        test_mask = torch.eq(test_node, 2)
        # test_em = self.token_embedding1(test_node.to(torch.int64))
        test_em = self.token_embedding2(test_node.to(torch.int64))
#        test_em = torch.cat([test_em, test_time.unsqueeze(-1).float(), test_coverage_weight.unsqueeze(-1).float(),
#                             test_jaccard.unsqueeze(-1).float(), test_per_coverage.unsqueeze(-1).float()], dim=-1)
        # test_em = torch.cat([test_em, test_coverage_weight.unsqueeze(-1).float()], dim=-1)
        test_em = torch.cat([test_em, test_time.unsqueeze(-1).float(), test_coverage_weight.unsqueeze(-1).float()], dim=-1)


        method_em = self.token_embedding(method_node.to(torch.int64))
        line_em = self.token_embedding(line_node.to(torch.int64))

        x = torch.cat([test_em, method_em, line_em], dim=1)

        for trans in self.transformerBlocks:
            x = trans.forward(x, test_mask, matrix)

        x = x[:, :test_node.size(1)]

        res_softmax = F.softmax(self.resLinear2(x).squeeze(-1).masked_fill(test_mask == 0, -1e9), dim=-1)

        # criterion = nn.L1Loss()
        # criterion = nn.NLLLoss()
        # criterion = nn.MSELoss()
        # loss = criterion(res_softmax, test_weight)

        loss = -torch.log(res_softmax.clamp(min=1e-10, max=1)) * test_weight

        loss = loss.sum(dim=-1)
        return loss, res_softmax, x
