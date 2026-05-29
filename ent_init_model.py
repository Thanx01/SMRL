import torch.nn as nn
import torch
import dgl
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class EntInit(nn.Module):
    def __init__(self, args):
        super(EntInit, self).__init__()
        self.args = args

        self.rel_head_emb = nn.Parameter(torch.Tensor(args.num_rel, args.ent_dim))
        self.rel_tail_emb = nn.Parameter(torch.Tensor(args.num_rel, args.ent_dim))

        nn.init.xavier_normal_(self.rel_head_emb, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_tail_emb, gain=nn.init.calculate_gain('relu'))

    def forward(self, g_bidir):
        num_edge = g_bidir.num_edges()
        etypes = g_bidir.edata['type']
        g_bidir.edata['ent_e'] = torch.zeros(num_edge, self.args.ent_dim).to(self.args.gpu)

        # 使用掩码来安全地索引
        rh_idx = etypes < self.args.num_rel
        rt_idx = etypes >= self.args.num_rel

        # 确保索引在合法范围内
        safe_rh_idx = etypes[rh_idx]
        safe_rt_idx = etypes[rt_idx] - self.args.num_rel
        
        # 对于rt_idx, 需要额外的检查来确保索引值不超出范围
        safe_rt_idx = torch.clamp(safe_rt_idx, 0, self.args.num_rel - 1)

        g_bidir.edata['ent_e'][rh_idx] = self.rel_head_emb[safe_rh_idx]
        g_bidir.edata['ent_e'][rt_idx] = self.rel_tail_emb[safe_rt_idx]

        message_func = dgl.function.copy_e('ent_e', 'msg')
        reduce_func = dgl.function.mean('msg', 'feat')
        g_bidir.update_all(message_func, reduce_func)
        g_bidir.edata.pop('ent_e')
