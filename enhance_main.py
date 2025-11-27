"""Unified interface to all dynamic graph model experiments"""
import math
import random
import sys
from tqdm import tqdm
import argparse
import os.path as osp
import h5py
import torch
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from utils import EarlyStopMonitor, RandEdgeSampler, load_subgraph_margin, get_item, get_item_edge, NeighborFinder
from models import *
from GraphM import GraphMixer
from TGN.tgn import TGN

# import wandb
# wandb.init(project="TempME", entity="enhance")

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def init_tensorboard(args):
    if args.wandb_sync == "disabled":
        return None
    
    # Create logs directory
    log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tensorboard_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique run name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.base_type}_{args.data}_{timestamp}"
    
    # Initialize the TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
    
    # Add hyperparameters
    writer.add_text("hyperparameters/model", args.base_type)
    writer.add_text("hyperparameters/dataset", args.data)
    writer.add_text("hyperparameters/learning_rate", str(args.lr))
    writer.add_text("hyperparameters/epochs", str(args.n_epoch))
    writer.add_text("hyperparameters/batch_size", str(args.bs))
    
    print(f"TensorBoard logging initialized at {log_dir}/{run_name}")
    return writer

degree_dict = {"wikipedia":20, "reddit":20 ,"uci":30 ,"mooc":60, "enron": 30, "enron_sampled": 30, "canparl": 30, "uslegis": 30, "uslegis_sampled": 30}
### Argument and global variables
parser = argparse.ArgumentParser('Motif Enhancement Verification')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument("--base_type", type=str, default="tgn", help="tgn or graphmixer or tgat")
parser.add_argument('--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=400, help='batch_size')
parser.add_argument('--test_bs', type=int, default=400, help='test batch_size')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=3, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--out_dim', type=int, default=32, help='number of attention dim')
parser.add_argument('--hid_dim', type=int, default=32, help='number of hidden dim')
parser.add_argument('--temp', type=float, default=0.07, help='temperature')
parser.add_argument('--num_layers', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--if_bern', type=bool, default=True, help='use bernoulli')
parser.add_argument('--save_model', type=bool, default=False, help='if save model')
parser.add_argument('--verbose', type=int, default=3, help='use dot product attention or mapping based')
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--lr_decay', type=float, default=0.999)
parser.add_argument('--task_type', type=str, default="motif-enhanced prediction")
parser.add_argument('--wandb_sync', type=str, default="online", help='online  or disabled')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


def norm_imp(imp):
    imp[imp < 0] = 0
    imp += 1e-16
    return imp / imp.sum()


### Load data and train val test split
def load_data(mode):
    g_df = pd.read_csv(osp.join(osp.dirname(osp.realpath(__file__)), 'processed/ml_{}.csv'.format(args.data)))
    val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    label_l = g_df.label.values
    ts_l = g_df.ts.values
    max_src_index = src_l.max()
    max_idx = max(src_l.max(), dst_l.max())
    random.seed(2023)
    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    num_total_unique_nodes = len(total_node_set)

    temp_val = list(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])))
    mask_node_set = set(random.sample(temp_val,
                                      int(0.1 * num_total_unique_nodes)))
    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
    valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)
    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time
    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    test_src_l = src_l[valid_test_flag]
    test_dst_l = dst_l[valid_test_flag]
    test_ts_l = ts_l[valid_test_flag]
    test_e_idx_l = e_idx_l[valid_test_flag]
    test_label_l = label_l[valid_test_flag]
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    train_ngh_finder = NeighborFinder(adj_list)
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = NeighborFinder(full_adj_list)
    train_rand_sampler = RandEdgeSampler((train_src_l,), (train_dst_l,))
    test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))
    if mode == "test":
        return test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, full_ngh_finder
    else:
        return train_rand_sampler, train_src_l, train_dst_l, train_ts_l, train_label_l, train_e_idx_l, train_ngh_finder



def eval_one_epoch(args, base_model, predictor, full_ngh_finder, src, dst, ts, val_e_idx_l, epoch, best_accuracy,
                   test_pack, test_edge,tb_writer):
    test_aps = []
    test_auc = []
    test_acc = []
    test_loss = []
    num_test_instance = len(src) - 1
    num_test_batch = math.ceil(num_test_instance / args.test_bs) - 1
    idx_list = np.arange(num_test_instance)
    criterion = torch.nn.BCEWithLogitsLoss()
    base_model.set_neighbor_sampler(full_ngh_finder)
    for k in tqdm(range(num_test_batch)):
        s_idx = k * args.test_bs
        e_idx = min(num_test_instance - 1, s_idx + args.test_bs)
        if s_idx == e_idx:
            continue
        batch_idx = idx_list[s_idx:e_idx]
        src_l_cut = src[batch_idx]
        dst_l_cut = dst[batch_idx]
        ts_l_cut = ts[batch_idx]
        e_l_cut = val_e_idx_l[batch_idx] if (val_e_idx_l is not None) else None
        subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(test_pack, batch_idx)
        edge_id_feature = get_item_edge(test_edge, batch_idx)
        predictor.eval()
        with torch.no_grad():
            #########################
            subgraph_src = base_model.grab_subgraph(src_l_cut, ts_l_cut)
            subgraph_tgt = base_model.grab_subgraph(dst_l_cut, ts_l_cut)
            subgraph_bgd = base_model.grab_subgraph(dst_l_fake, ts_l_cut)
            #########################
            src_emb, tgt_emb, bgd_emb = base_model.get_node_emb(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                               subgraph_src, subgraph_tgt, subgraph_bgd)
            pos_logit, neg_logit = predictor.enhance_predict_agg(ts_l_cut, walks_src, walks_tgt, walks_bgd,
                                                               edge_id_feature, src_emb, tgt_emb, bgd_emb)
            size = len(src_l_cut)
            pos_label = torch.ones((size, 1), dtype=torch.float, device=args.device, requires_grad=False)
            neg_label = torch.zeros((size, 1), dtype=torch.float, device=args.device, requires_grad=False)
            loss = criterion(pos_logit, pos_label) + criterion(neg_logit, neg_label)
            pos_prob = pos_logit.sigmoid().squeeze(-1)
            neg_prob = neg_logit.sigmoid().squeeze(-1)
            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5

            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            test_aps.append(average_precision_score(true_label, pred_score))
            test_auc.append(roc_auc_score(true_label, pred_score))
            test_acc.append((pred_label == true_label).mean())
            test_loss.append(loss.item())

    aps_epoch = np.mean(test_aps)
    auc_epoch = np.mean(test_auc)
    acc_epoch = np.mean(test_acc)
    loss_epoch = np.mean(test_loss)

 # Add feature importance visualization
    if tb_writer is not None and epoch % 5 == 0 and 'edge_id_feature' in locals():
        # Visualize edge feature importance
        try:
            # Get the importance weights from the predictor if available
            if hasattr(predictor, 'edge_weights') and predictor.edge_weights is not None:
                weights = predictor.edge_weights.detach().cpu().numpy()
                feature_names = [f"Feature_{i}" for i in range(weights.shape[0])]
                
                # Create a bar chart
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.bar(feature_names, weights)
                plt.title(f"Edge Feature Importance (Epoch {epoch})")
                plt.ylabel("Importance")
                plt.xticks(rotation=45)
                
                # Add to TensorBoard
                tb_writer.add_figure("Model/EdgeFeatureImportance", plt.gcf(), epoch)
                plt.close()
        except Exception as e:
            print(f"Warning: Could not visualize feature importance: {e}")

    wandb_dict = {'Test Loss': loss_epoch, "Test Aps": aps_epoch, "Test Auc": auc_epoch, 'Test Acc': acc_epoch}

    if tb_writer is not None:
        tb_writer.add_scalar('Test/Loss', loss_epoch, epoch)
        tb_writer.add_scalar('Test/Aps', aps_epoch, epoch)
        tb_writer.add_scalar('Test/Auc', auc_epoch, epoch)
        tb_writer.add_scalar('Test/Acc', acc_epoch, epoch)
    print((f'Testing Epoch: {epoch} | '
           f'Testing loss: {loss_epoch} | '
           f'Testing Aps: {aps_epoch} | '
           f'Testing Auc: {auc_epoch} | '
           f'Testing Acc: {acc_epoch} | '))

    if aps_epoch > best_accuracy:
        if args.save_model:
            model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'params', f'predictors/{args.base_type}/')
            if not osp.exists(model_path):
                os.makedirs(model_path)
            save_path = f"{args.data}.pt"
            torch.save(predictor, osp.join(model_path, save_path))
            print(f"Save model to {osp.join(model_path, save_path)}")

        # Add embeddings visualization
        if tb_writer is not None and epoch % 10 == 0:  # Every 10 epochs
            try:
                # Get a sample of node embeddings
                with torch.no_grad():
                    sample_size = min(100, len(src))
                    sample_idx = np.random.choice(len(src), sample_size, replace=False)
                    sample_src = src[sample_idx]
                    sample_ts = ts[sample_idx]
                    
                    # Get embeddings using the base model
                    base_model.eval()
                    sample_embs = base_model.compute_temporal_embeddings(sample_src, sample_ts)
                    
                    # Add embeddings to TensorBoard
                    metadata = [f"Node_{int(n)}" for n in sample_src]
                    tb_writer.add_embedding(
                        mat=sample_embs.detach().cpu(),
                        metadata=metadata,
                        global_step=epoch,
                        tag=f"node_embeddings"
                    )
            except Exception as e:
                print(f"Warning: Could not visualize embeddings: {e}")

        return aps_epoch
    else:
        return best_accuracy


def train(args, base_model, train_pack, test_pack, train_edge, test_edge, tb_writer):
    if args.base_type == "tgat":
        predictor = TempME_TGAT(base_model, data=args.data, out_dim=args.out_dim, hid_dim=args.hid_dim, temp=args.temp,
                                dropout_p=args.drop_out, device=args.device)
    else:
        predictor = TempME(base_model, base_model_type=args.base_type, data=args.data, out_dim=args.out_dim, hid_dim=args.hid_dim,
                                temp=args.temp, if_cat_feature=True, dropout_p=args.drop_out, device=args.device)
    predictor = predictor.to(args.device)
# Add model graph visualization
    if tb_writer is not None:
        try:
            # Create dummy inputs for the model
            batch_size = 2
            dummy_src = torch.zeros((batch_size,), dtype=torch.long, device=args.device)
            dummy_dst = torch.zeros((batch_size,), dtype=torch.long, device=args.device)
            dummy_ts = torch.zeros((batch_size,), dtype=torch.float, device=args.device)
            dummy_idx = torch.zeros((batch_size,), dtype=torch.long, device=args.device)
            
            # Create dummy subgraph and walks data
            dummy_subgraph = [torch.zeros((batch_size, 3), dtype=torch.long, device=args.device)]
            dummy_walks = torch.zeros((batch_size, 3, 3), dtype=torch.long, device=args.device)
            dummy_edge = torch.zeros((batch_size, 3), dtype=torch.long, device=args.device)
            
            # Try to trace the model (may not work with all model architectures)
            tb_writer.add_graph(predictor, [dummy_ts, dummy_walks, dummy_walks, dummy_walks, 
                                        dummy_edge, dummy_src, dummy_dst, dummy_dst])
        except Exception as e:
            print(f"Warning: Could not add model graph to TensorBoard: {e}")

    optimizer = torch.optim.Adam(list(predictor.parameters()) + list(base_model.parameters()),
                                 lr=args.lr,
                                 betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    rand_sampler, src_l, dst_l, ts_l, label_l, e_idx_l, ngh_finder = load_data(mode="training")
    test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, full_ngh_finder = load_data(
        mode="test")
    num_instance = len(src_l) - 1
    num_batch = math.ceil(num_instance / args.bs)
    best_acc = 0
    print(f"start training: {args.data}")
    print('num of training instances: {}'.format(num_instance))
    print('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    np.random.shuffle(idx_list)
    for epoch in range(args.n_epoch):
        base_model.set_neighbor_sampler(ngh_finder)
        train_aps = []
        train_auc = []
        train_acc = []
        train_loss = []
        np.random.shuffle(idx_list)
        predictor.train()
        base_model.train()
        for k in tqdm(range(num_batch)):
            s_idx = k * args.bs
            e_idx = min(num_instance - 1, s_idx + args.bs)
            if s_idx == e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]
            src_l_cut, dst_l_cut = src_l[batch_idx], dst_l[batch_idx]
            ts_l_cut = ts_l[batch_idx]
            e_l_cut = e_idx_l[batch_idx]
            subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(train_pack,
                                                                                                             batch_idx)
            edge_id_feature = get_item_edge(train_edge, batch_idx)
            optimizer.zero_grad()
            #########################
            subgraph_src = base_model.grab_subgraph(src_l_cut, ts_l_cut)
            subgraph_tgt = base_model.grab_subgraph(dst_l_cut, ts_l_cut)
            subgraph_bgd = base_model.grab_subgraph(dst_l_fake, ts_l_cut)
            #########################
            src_emb, tgt_emb, bgd_emb = base_model.get_node_emb(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                                subgraph_src, subgraph_tgt, subgraph_bgd)
            pos_logit, neg_logit = predictor.enhance_predict_agg(ts_l_cut, walks_src, walks_tgt, walks_bgd,
                                                                 edge_id_feature, src_emb, tgt_emb, bgd_emb)
            size = len(src_l_cut)
            pos_label = torch.ones((size, 1), dtype=torch.float, device=args.device, requires_grad=False)
            neg_label = torch.zeros((size, 1), dtype=torch.float, device=args.device, requires_grad=False)
            loss = criterion(pos_logit, pos_label) + criterion(neg_logit, neg_label)
            loss.backward()
            optimizer.step()
            
# Add learning rate tracking for TensorBoard
            if tb_writer is not None:
                current_lr = optimizer.param_groups[0]['lr']
                tb_writer.add_scalar('Train/LearningRate', current_lr, epoch)

            if args.base_type == "tgn":
                base_model.memory.detach_memory()

            with torch.no_grad():
                pos_prob = pos_logit.sigmoid().squeeze(-1)
                neg_prob = neg_logit.sigmoid().squeeze(-1)
                pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                train_aps.append(average_precision_score(true_label, pred_score))
                train_auc.append(roc_auc_score(true_label, pred_score))
                train_acc.append((pred_label == true_label).mean())
                train_loss.append(loss.item())

        aps_epoch = np.mean(train_aps)
        auc_epoch = np.mean(train_auc)
        acc_epoch = np.mean(train_acc)
        loss_epoch = np.mean(train_loss)
        wandb_dict = {'Train Loss': loss_epoch, "Train Aps": aps_epoch, "Train Auc": auc_epoch, 'Train Acc': acc_epoch}
        if tb_writer is not None:
            tb_writer.add_scalar('Train/Loss', loss_epoch, epoch)
            tb_writer.add_scalar('Train/Aps', aps_epoch, epoch)
            tb_writer.add_scalar('Train/Auc', auc_epoch, epoch)
            tb_writer.add_scalar('Train/Acc', acc_epoch, epoch)
        print((f'Training Epoch: {epoch} | '
               f'Training loss: {loss_epoch} | '
               f'Training Aps: {aps_epoch} | '
               f'Training Auc: {auc_epoch} | '
               f'Training Acc: {acc_epoch} | '))

        ### evaluation:
        if (epoch + 1) % args.verbose == 0:
            if args.base_type == "tgn":
                train_memory_backup = base_model.memory.backup_memory()
            best_acc = eval_one_epoch(args, base_model, predictor, full_ngh_finder, test_src_l,
                                      test_dst_l, test_ts_l, test_e_idx_l, epoch, best_acc, test_pack, test_edge,tb_writer)
            if args.base_type == "tgn":
                base_model.memory.restore_memory(train_memory_backup)

if __name__ == '__main__':
    if torch.cuda.is_available():
        # First check how many GPUs are available and print their information
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} CUDA-capable GPU(s)")
        
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Always use GPU 0 for the NVIDIA card
        args.gpu = 0
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(f'cuda:{args.gpu}')
        print(f"CUDA is available. Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        # Print memory information
        print(f"GPU memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.2f} GB")
    else:
        args.device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")

    # Initialize TensorBoard writer
    tb_writer = init_tensorboard(args)

    args.n_degree = degree_dict[args.data]
    gnn_model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'params', 'tgnn',
                              f'{args.base_type}_{args.data}.pt')
    base_model = torch.load(gnn_model_path, weights_only=False).to(args.device)
    pre_load_train = h5py.File(osp.join(osp.dirname(osp.realpath(__file__)), 'processed', f'{args.data}_train_cat.h5'),
                               'r')
    pre_load_test = h5py.File(osp.join(osp.dirname(osp.realpath(__file__)), 'processed', f'{args.data}_test_cat.h5'),
                              'r')
    e_feat = np.load(osp.join(osp.dirname(osp.realpath(__file__)), 'processed', f'ml_{args.data}.npy'))
    n_feat = np.load(osp.join(osp.dirname(osp.realpath(__file__)), 'processed', f'ml_{args.data}_node.npy'))

    train_pack = load_subgraph_margin(args, pre_load_train)
    test_pack = load_subgraph_margin(args, pre_load_test)

    train_edge = np.load(osp.join(osp.dirname(osp.realpath(__file__)), 'processed', f'{args.data}_train_edge.npy'))
    test_edge = np.load(osp.join(osp.dirname(osp.realpath(__file__)), 'processed', f'{args.data}_test_edge.npy'))

    try:
        # Pass tb_writer to train
        train(args, base_model, train_pack=train_pack, test_pack=test_pack, train_edge=train_edge, test_edge=test_edge, tb_writer=tb_writer)
    finally:
        # Close TensorBoard writer
        if tb_writer is not None:
            tb_writer.close()






