import torch
import torch.nn as nn
import dgl


class EntInit(nn.Module):
    def __init__(self, args):
        super(EntInit, self).__init__()
        self.args = args

        self.rel_head_emb = nn.Parameter(torch.Tensor(args.num_rel, args.ent_dim))
        self.rel_tail_emb = nn.Parameter(torch.Tensor(args.num_rel, args.ent_dim))
        self.attr_padding_idx = args.num_attr
        self.attr_emb = nn.Embedding(args.num_attr + 1, args.ent_dim, padding_idx=self.attr_padding_idx)

        nn.init.xavier_normal_(self.rel_head_emb, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_tail_emb, gain=nn.init.calculate_gain('relu'))
        if args.num_attr > 0:
            nn.init.xavier_normal_(self.attr_emb.weight[:-1], gain=nn.init.calculate_gain('relu'))
        with torch.no_grad():
            self.attr_emb.weight[self.attr_padding_idx].fill_(0)

    def forward(self, g_bidir):
        num_edge = g_bidir.num_edges()
        etypes = g_bidir.edata['type']
        device = self.rel_head_emb.device
        g_bidir.edata['ent_e'] = torch.zeros(num_edge, self.args.ent_dim, device=device)

        rel_head_idx = etypes < self.args.num_rel
        rel_tail_idx = etypes >= self.args.num_rel

        g_bidir.edata['ent_e'][rel_head_idx] = self.rel_head_emb[etypes[rel_head_idx]]
        tail_indices = etypes[rel_tail_idx] - self.args.num_rel
        tail_indices = torch.clamp(tail_indices, 0, self.args.num_rel - 1)
        g_bidir.edata['ent_e'][rel_tail_idx] = self.rel_tail_emb[tail_indices]

        message_func = dgl.function.copy_e('ent_e', 'msg')
        reduce_func = dgl.function.mean('msg', 'feat')
        g_bidir.update_all(message_func, reduce_func)
        g_bidir.edata.pop('ent_e')

        if getattr(self.args, 'use_attr', True) and 'attr_ids' in g_bidir.ndata:
            attr_ids = g_bidir.ndata['attr_ids'].to(device)
            attr_ids = torch.clamp(attr_ids, 0, self.attr_padding_idx)
            attr_mask = g_bidir.ndata.get('attr_mask')
            if attr_mask is None:
                attr_mask = (attr_ids != self.attr_padding_idx).float()
            attr_mask = attr_mask.to(device).unsqueeze(-1)

            attr_feat = self.attr_emb(attr_ids)
            attr_feat = (attr_feat * attr_mask).sum(dim=1)
            attr_count = attr_mask.sum(dim=1).clamp(min=1.0)
            attr_feat = attr_feat / attr_count
            g_bidir.ndata['feat'] = g_bidir.ndata['feat'] + getattr(self.args, 'attr_weight', 1.0) * attr_feat