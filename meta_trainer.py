from utils import get_g_bidir
from datasets import TrainSubgraphDataset, ValidSubgraphDataset
from torch.utils.data import DataLoader
import torch
from torch import optim
from trainer import Trainer
import dgl
from collections import defaultdict as ddict


class MetaTrainer(Trainer):
    def __init__(self, args):
        super(MetaTrainer, self).__init__(args)
        # dataloader
        train_subgraph_dataset = TrainSubgraphDataset(args)
        valid_subgraph_dataset = ValidSubgraphDataset(args)
        self.train_subgraph_dataloader = DataLoader(train_subgraph_dataset, batch_size=args.metatrain_bs,
                                                    shuffle=True, collate_fn=TrainSubgraphDataset.collate_fn)
        self.valid_subgraph_dataloader = DataLoader(valid_subgraph_dataset, batch_size=args.metatrain_bs,
                                                    shuffle=False, collate_fn=ValidSubgraphDataset.collate_fn)

        # optim
        self.optimizer = optim.Adam(list(self.ent_init.parameters()) + list(self.rgcn.parameters())
                                    + list(self.kge_model.parameters()), lr=self.args.metatrain_lr)

    def load_pretrain(self):
        state = torch.load(self.args.pretrain_state, map_location=self.args.gpu)
        self.ent_init.load_state_dict(state['ent_init'])
        self.rgcn.load_state_dict(state['rgcn'])
        self.kge_model.load_state_dict(state['kge_model'])


    # def train(self):
    #     step = 0
    #     best_step = 0
    #     best_eval_rst = {'mrr': 0, 'hits@1': 0, 'hits@5': 0, 'hits@10': 0}
    #     bad_count = 0
    #     self.logger.info('start meta-training')

        # for e in range(self.args.metatrain_num_epoch):
        #     print('self.train_subgraph_dataloader:', self.train_subgraph_dataloader)
        #     for batch in self.train_subgraph_dataloader:
        #         batch_loss = 0
        #         classification_loss = 0  # 初始化分类损失

        #         batch_sup_g = dgl.batch([get_g_bidir(d[0], self.args) for d in batch]).to(self.args.gpu)
        #         ent_emb, class_logits = self.get_ent_emb_and_logits(batch_sup_g)  # 获取实体嵌入和分类logits
        #         class_labels = batch[1]  # 假设batch[1]是类别标签

        #         # 计算分类损失
        #         classification_loss = F.nll_loss(class_logits, class_labels.to(self.args.gpu))

        #         sup_g_list = dgl.unbatch(batch_sup_g)
        #         for batch_i, data in enumerate(batch):
        #             que_tri, que_neg_tail_ent, que_neg_head_ent = [d.to(self.args.gpu) for d in data[2:5]]  # 调整数据索引
        #             ent_emb = sup_g_list[batch_i].ndata['h']
        #             # KGE损失
        #             loss = self.get_loss(que_tri, que_neg_tail_ent, que_neg_head_ent, ent_emb)
        #             batch_loss += loss

        #         total_loss = (batch_loss + classification_loss) / len(batch)  # 合并损失
        #         self.optimizer.zero_grad()
        #         total_loss.backward()
        #         self.optimizer.step()

        #         step += 1
        #         self.logger.info('step: {} | loss: {:.4f}'.format(step, total_loss.item()))
        #         self.write_training_loss(batch_loss.item(), classification_loss.item(), step)  # 记录损失

        #         if step % self.args.metatrain_check_per_step == 0:
        #             eval_res = self.evaluate_valid_subgraphs()
        #             self.write_evaluation_result(eval_res, step)

        #             if eval_res['mrr'] > best_eval_rst['mrr']:
        #                 best_eval_rst = eval_res
        #                 best_step = step
        #                 self.logger.info('best model | mrr {:.4f}'.format(best_eval_rst['mrr']))
        #                 self.save_checkpoint(step)
        #                 bad_count = 0
        #             else:
        #                 bad_count += 1
        #                 self.logger.info('best model is at step {0}, mrr {1:.4f}, bad count {2}'.format(
        #                     best_step, best_eval_rst['mrr'], bad_count))

        # self.logger.info('finish meta-training')
        # self.logger.info('save best model')
        # self.save_model(best_step)

        # self.logger.info('best validation | mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
        #     best_eval_rst['mrr'], best_eval_rst['hits@1'],
        #     best_eval_rst['hits@5'], best_eval_rst['hits@10']))

        # #这里self.indtest_train_g是用于评估阶段的图数据。通过这种方式，EntInit为评估阶段的图数据中的实体生成初始嵌入。
        # self.before_test_load()
        # self.evaluate_indtest_test_triples(num_cand=50)
    def train(self):
        step = 0
        best_step = 0
        best_eval_rst = {'mrr': 0, 'hits@1': 0, 'hits@5': 0, 'hits@10': 0}
        bad_count = 0
        self.logger.info('start meta-training')

        for e in range(self.args.metatrain_num_epoch):
            print('self.train_subgraph_dataloader:',self.train_subgraph_dataloader)
            for batch in self.train_subgraph_dataloader:
                batch_loss = 0

                batch_sup_g = dgl.batch([get_g_bidir(d[0], self.args, d[1]) for d in batch]).to(self.args.gpu)
                #这里的batch_sup_g是由支持集（support set）中的图数据组成的批次。这个步骤发生在元训练的每个批次中，其目的是为这些支持集图中的实体生成初始嵌入，以便后续训练步骤使用。
                self.get_ent_emb(batch_sup_g)
                sup_g_list = dgl.unbatch(batch_sup_g)
                for batch_i, data in enumerate(batch):
                    que_tri, que_neg_tail_ent, que_neg_head_ent = [d.to(self.args.gpu) for d in data[2:]]
                    ent_emb = sup_g_list[batch_i].ndata['h']
                    # kge loss
                    loss = self.get_loss(que_tri, que_neg_tail_ent, que_neg_head_ent, ent_emb)

                    batch_loss += loss

                batch_loss /= len(batch)
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                step += 1
                self.logger.info('step: {} | loss: {:.4f}'.format(step, batch_loss.item()))
                self.write_training_loss(batch_loss.item(), step)

                if step % self.args.metatrain_check_per_step == 0:
                    eval_res = self.evaluate_valid_subgraphs()
                    self.write_evaluation_result(eval_res, step)

                    if eval_res['mrr'] > best_eval_rst['mrr']:
                        best_eval_rst = eval_res
                        best_step = step
                        self.logger.info('best model | mrr {:.4f}'.format(best_eval_rst['mrr']))
                        self.save_checkpoint(step)
                        bad_count = 0
                    else:
                        bad_count += 1
                        self.logger.info('best model is at step {0}, mrr {1:.4f}, bad count {2}'.format(
                            best_step, best_eval_rst['mrr'], bad_count))

        self.logger.info('finish meta-training')
        self.logger.info('save best model')
        self.save_model(best_step)

        self.logger.info('best validation | mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            best_eval_rst['mrr'], best_eval_rst['hits@1'],
            best_eval_rst['hits@5'], best_eval_rst['hits@10']))

        self.before_test_load()
        self.evaluate_indtest_test_triples(num_cand=50)




    def evaluate_valid_subgraphs(self):
        all_results = ddict(int)
        for batch in self.valid_subgraph_dataloader:
            batch_sup_g = dgl.batch([get_g_bidir(d[0], self.args, d[1]) for d in batch]).to(self.args.gpu)
            self.get_ent_emb(batch_sup_g)
            sup_g_list = dgl.unbatch(batch_sup_g)

            for batch_i, data in enumerate(batch):
                que_dataloader = data[2]
                ent_emb = sup_g_list[batch_i].ndata['h']

                results = self.evaluate(ent_emb, que_dataloader)

                for k, v in results.items():
                    all_results[k] += v

        for k, v in all_results.items():
            all_results[k] = v / self.args.num_valid_subgraph

        self.logger.info('valid on valid subgraphs')
        self.logger.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            all_results['mrr'], all_results['hits@1'],
            all_results['hits@5'], all_results['hits@10']))

        return all_results