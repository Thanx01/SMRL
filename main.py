import argparse
import pickle
from utils import init_dir, set_seed, get_num_rel, get_num_attr
from meta_trainer import MetaTrainer
from post_trainer import PostTrainer
import os
from subgraph import gen_subgraph_datasets
from pre_process import data2pkl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='region_v2')
    parser.add_argument('--ind_data_name', default=None, type=str,
                        help='Name of the unseen/inductive dataset folder. If omitted, it is inferred from data_name.')

    parser.add_argument('--num_attr', type=int, default=None,
                        help='Number of attributes. Defaults to the value detected during preprocessing.')
    parser.add_argument('--use_attr', nargs='?', const=True,
                        type=lambda x: str(x).lower() not in ['0', 'false', 'no'],
                        default=True, help='Whether to use attribute aggregation.')
    parser.add_argument('--attr_weight', type=float, default=1.0,
                        help='Weight applied to the aggregated attribute embedding.')
    parser.add_argument('--force_preprocess', action='store_true',
                        help='Regenerate the cached pickle data even if it already exists.')


    parser.add_argument('--name', default='region_v2_transe', type=str)

    parser.add_argument('--step', default='fine_tune', type=str, choices=['meta_train', 'fine_tune'])
    parser.add_argument('--metatrain_state', default='./state/region_v2_rotate/region_v2_transe.best', type=str)

    parser.add_argument('--state_dir', '-state_dir', default='./state', type=str)
    parser.add_argument('--log_dir', '-log_dir', default='./log', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str)

    # params for subgraph
    parser.add_argument('--num_train_subgraph', default=10000, type=int)
    parser.add_argument('--num_valid_subgraph', default=200, type=int)
    parser.add_argument('--num_sample_for_estimate_size', default=50, type=int)
    parser.add_argument('--rw_0', default=10, type=int)
    parser.add_argument('--rw_1', default=10, type=int)
    parser.add_argument('--rw_2', default=5, type=int)
    parser.add_argument('--min_subgraph_nodes', default=50, type=int)
    parser.add_argument('--num_sample_cand', default=5, type=int)

    # params for meta-train
    parser.add_argument('--metatrain_num_neg', default=32, type=int)
    parser.add_argument('--metatrain_num_epoch', default=1, type=int)
    # parser.add_argument('--metatrain_bs', default=64, type=int)
    parser.add_argument('--metatrain_bs', default=8, type=int)

    parser.add_argument('--metatrain_lr', default=0.01, type=float)
    parser.add_argument('--metatrain_check_per_step', default=10, type=int)#check_step
    parser.add_argument('--indtest_eval_bs', default=512, type=int)

    # params for fine-tune
    parser.add_argument('--posttrain_num_neg', default=64, type=int)
    parser.add_argument('--posttrain_bs', default=512, type=int)
    parser.add_argument('--posttrain_lr', default=0.001, type=float)
    parser.add_argument('--posttrain_num_epoch', default=10, type=int)
    parser.add_argument('--posttrain_check_per_epoch', default=1, type=int)

    # params for R-GCN
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--num_bases', default=4, type=int)
    parser.add_argument('--emb_dim', default=32, type=int)

    # params for KGE
    parser.add_argument('--kge', default='TransE', type=str, choices=['TransE', 'DistMult', 'ComplEx', 'RotatE'])
    parser.add_argument('--gamma', default=10, type=float)
    parser.add_argument('--adv_temp', default=1, type=float)

    parser.add_argument('--gpu', default='cuda:2', type=str)
    parser.add_argument('--seed', default=1234, type=int)

    args = parser.parse_args()
    cli_num_attr = args.num_attr
    init_dir(args)

    args.ent_dim = args.emb_dim
    args.rel_dim = args.emb_dim
    if args.kge in ['ComplEx', 'RotatE']:
        args.ent_dim = args.emb_dim * 2
    if args.kge in ['ComplEx']:
        args.rel_dim = args.emb_dim * 2

    # specify the paths for original data and subgraph db
    args.data_path = f'./data/{args.data_name}.pkl'
    db_suffix = 'attr_subgraph' if args.use_attr else 'subgraph'
    args.db_path = f'./data/{args.data_name}_{db_suffix}'

    # load original data and make index
    if args.force_preprocess or not os.path.exists(args.data_path):
        data2pkl(args.data_name, ind_data_name=args.ind_data_name)
    else:
        cached_data = pickle.load(open(args.data_path, 'rb'))
        if 'num_attr' not in cached_data:
            data2pkl(args.data_name, ind_data_name=args.ind_data_name)

    if cli_num_attr is None:
        args.num_attr = get_num_attr(args)

    print(args.db_path)
    if not os.path.exists(args.db_path):
        gen_subgraph_datasets(args)
    # gen_subgraph_datasets(args)

    args.num_rel = get_num_rel(args)

    set_seed(args.seed)

    # if args.step == 'meta_train':
    #     meta_trainer = MetaTrainer(args)
    #     meta_trainer.train()
    # elif args.step == 'fine_tune':
    #     post_trainer = PostTrainer(args)
    #     post_trainer.train()

    if args.step == 'meta_train':
        meta_trainer = MetaTrainer(args)
        meta_trainer.train()
    elif args.step == 'fine_tune':
        post_trainer = PostTrainer(args)
        post_trainer.train()
