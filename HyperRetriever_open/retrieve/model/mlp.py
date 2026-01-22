import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, pred_in_size, emb_size):
        """
        SubgraphRAG官方MLP架构
        
        Args:
            pred_in_size: 输入特征总维度 [question || head || relation || tail || dde]
            emb_size: 第一层隐藏层维度
        """
        super().__init__()
        
        # 第一层：输入 -> 隐藏层
        self.linear1 = nn.Linear(pred_in_size, emb_size)
        
        # 激活函数
        self.relu = nn.ReLU()
        
        # 第二层：隐藏层 -> 输出 (单个数值)
        self.linear2 = nn.Linear(emb_size, 1)
    
    def forward(self, x):
        """
        前向传播
        x: [batch_size, pred_in_size] - 打包的特征
        返回: [batch_size, 1] - 三元组相关性分数
        """
        x = self.linear1(x)        # [batch_size, emb_size]
        x = self.relu(x)           # [batch_size, emb_size] 
        x = self.linear2(x)        # [batch_size, 1]
        return x
