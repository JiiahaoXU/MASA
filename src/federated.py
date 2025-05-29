import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from agent_sparse import Agent as Agent_s
from aggregation import Aggregation
import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
import logging
import time
import argparse
from shutil import copyfile
import os

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='pass in a parameter')
    
    parser.add_argument('--data', type=str, default='cifar10',
                        help="dataset we want to train on")
    
    parser.add_argument('--num_agents', type=int, default=20,
                        help="number of agents")
    
    parser.add_argument('--agent_frac', type=float, default=1.0,
                        help="fraction of agents per round")
    
    parser.add_argument('--num_corrupt', type=int, default=2,
                        help="number of corrupt agents")
    
    parser.add_argument('--rounds', type=int, default=100,
                        help="number of communication rounds")
    
    parser.add_argument('--local_ep', type=int, default=2,
                        help="number of local epochs")
    
    parser.add_argument('--attacker_local_ep', type=int, default=2,
                        help="number of local epochs")
    
    parser.add_argument('--bs', type=int, default=64,
                        help="local batch size: B")
    
    parser.add_argument('--client_lr', type=float, default=0.1,
                        help='clients learning rate')
    parser.add_argument('--malicious_client_lr', type=float, default=0.1,
                        help='clients learning rate')
    parser.add_argument('--server_lr', type=float, default=1,
                        help='servers learning rate for signSGD')
    
    parser.add_argument('--target_class', type=int, default=7,
                        help="target class for backdoor attack")
    
    parser.add_argument('--poison_frac', type=float, default=0.5,
                        help="fraction of dataset to corrupt for backdoor attack")
    
    parser.add_argument('--pattern_type', type=str, default='plus',
                        help="shape of bd pattern")
    parser.add_argument('--theta', type=int, default=8,
                        help="break ties when votes sum to 0")
    parser.add_argument('--theta_ld', type=int, default=10,
                        help="break ties when votes sum to 0")
    parser.add_argument('--snap', type=int, default=1,
                        help="do inference in every num of snap rounds")
    parser.add_argument('--device',  default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help="To use cuda, set to a specific GPU ID.")
    parser.add_argument('--num_workers', type=int, default=0, 
                        help="num of workers for multithreading")
    parser.add_argument('--dense_ratio', type=float, default=0.4,
                        help="dense ratios")
    parser.add_argument('--anneal_factor', type=float, default=0.0001)
    # parser.add_argument('--se_threshold', type=float, default=1e-4)
    parser.add_argument('--non_iid', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--alpha',type=float, default=0.5)
    parser.add_argument('--attack',type=str, default="badnet", choices=['badnet', 'DBA', 'neurotoxin', 'pgd', 'lie'])
    parser.add_argument('--aggr', type=str, default='avg', choices=['avg', 'masa', 'rlr', 'mkrum', 'mul_metric', 'lockdown', 'fg', 'rfa'],
                        help="aggregation function to aggregate agents' local weights")
    parser.add_argument('--lr_decay',type=float, default=0.99)
    parser.add_argument('--momentum',type=float, default=0.0)
    parser.add_argument('--mask_init', type=str, default="ERK")
    parser.add_argument('--dis_check_gradient', action='store_true', default=False)
    parser.add_argument('--wd', type=float, default= 1e-4)
    parser.add_argument('--same_mask', type=int, default= 1)
    parser.add_argument('--exp_name_extra', type=str, help='defence name', default='')
    parser.add_argument('--super_power', action='store_true')
    parser.add_argument('--clean', action='store_true')

    parser.add_argument('--fusion_lambda', type=float, default=0.7)

    parser.add_argument('--filter_delta', type=float, default=1.0)
    
    parser.add_argument('--unlearn_lr', type=float, default=0.001)
    parser.add_argument('--unlearn_moment', type=float, default=0.9)

    args = parser.parse_args()

    # Handle the log name
    if args.data == 'cifar100':
        args.rounds = 150

    if args.clean:
        args.num_corrupt = 0
        args.exp_name_extra = 'clean'

    if args.super_power:
        args.exp_name_extra = 'sp'

    if args.momentum != 0.0:
        args.exp_name_extra = 'mom_%.3f' % (args.momentum)

    if args.aggr == 'lockdown' and args.attack == 'neurotoxin':
        args.theta_ld = 15
    
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    if not args.debug:
        logPath = "logs"
        time_str = time.strftime("%Y-%m-%d-%H-%M")

        if args.non_iid:
            iid_str = 'noniid(%.1f)' % args.alpha
        else:
            iid_str = 'iid'

        args.exp_name = iid_str + '_pr(%.1f)' % args.poison_frac

        if args.exp_name_extra != '':
            args.exp_name += '_%s' % args.exp_name_extra

        fileName = "%s_%s" % (time_str, args.exp_name)

        dir_path = '%s/%s/attack_%s_ar_%.2f/defense_%s/%s/' % (logPath, args.data, args.attack, args.num_corrupt / args.num_agents, args.aggr, fileName)
        file_path = dir_path + 'backup_file/'

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        backup_file = ['aggregation.py', 'federated.py', 'agent.py']

        for file in backup_file:
            copyfile('./%s' % file, file_path + file)

        fileHandler = logging.FileHandler("{0}/{1}.log".format(dir_path, fileName))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG) 
        console_handler.setFormatter(logFormatter)
        rootLogger.addHandler(console_handler)
    logging.info(args)


    train_dataset, val_dataset, server_dataset = utils.get_datasets(args.data)

    if args.data == "cifar100":
        num_target = 100
    else:
        num_target = 10

    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)
    if args.non_iid:
        user_groups = utils.distribute_data_dirichlet(train_dataset, args)
    else:
        user_groups = utils.distribute_data(train_dataset, args, n_classes=num_target)

    idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()

    poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)

    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                                    pin_memory=False)
    
    idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
    poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    utils.poison_dataset(poisoned_val_set_only_x.dataset, args, idxs, poison_all=True, modify_label=False)

    poisoned_val_only_x_loader = DataLoader(poisoned_val_set_only_x, batch_size=args.bs, shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=False)

    # initialize a model, and the agents
    global_model = models.get_model(args.data).to(args.device)

    global_mask = {}
    neurotoxin_mask = {}
    updates_dict = {}
    n_model_params = len(parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]))
    params = {name: copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()}

    if args.aggr == "lockdown":
        sparsity = utils.calculate_sparsities(args, params, distribution=args.mask_init)
        mask = utils.init_masks(params, sparsity)

    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        if args.aggr == "lockdown":
            if args.same_mask==0:
                agent = Agent_s(_id, args, train_dataset, user_groups[_id], mask=utils.init_masks(params, sparsity))
            else:
                agent = Agent_s(_id, args, train_dataset, user_groups[_id], mask=mask)
        else:
            agent = Agent(_id, args, train_dataset, user_groups[_id])
        agent.is_malicious = 1 if _id < args.num_corrupt else 0
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

        logging.info('build client:{} mal:{} data_num:{}'.format(_id, agent.is_malicious, agent.n_data))

    aggregator = Aggregation(agent_data_sizes, n_model_params, poisoned_val_loader, args, None, server_dataset)

    criterion = nn.CrossEntropyLoss().to(args.device)
    agent_updates_list = []
    worker_id_list = []
    agent_updates_dict = {}
    mask_aggrement = []

    best_acc = -1

    for rnd in range(1, args.rounds + 1):
        logging.info("--------round {} ------------".format(rnd))
        # mask = torch.ones(n_model_params)
        rnd_global_params = parameters_to_vector([ copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()])
        agent_updates_dict = {}
        chosen = np.random.choice(args.num_agents, math.floor(args.num_agents * args.agent_frac), replace=False)
        chosen = sorted(chosen)
        if args.aggr == "lockdown" or args.aggr == "fedimp":
            old_mask = [copy.deepcopy(agent.mask) for agent in agents]

        for agent_id in chosen:
            if agents[agent_id].is_malicious and args.super_power:
                continue
            # logging.info(torch.sum(rnd_global_params))
            global_model = global_model.to(args.device)

            if args.aggr == "lockdown":
                update = agents[agent_id].local_train(global_model, criterion, rnd, global_mask=global_mask, neurotoxin_mask = neurotoxin_mask, updates_dict=updates_dict)
            else:
                update = agents[agent_id].local_train(global_model, criterion, rnd, neurotoxin_mask=neurotoxin_mask)
            agent_updates_dict[agent_id] = update
            utils.vector_to_model(copy.deepcopy(rnd_global_params), global_model)

        # aggregate params obtained by agents and update the global params

        updates_dict, neurotoxin_mask = aggregator.aggregate_updates(global_model, agent_updates_dict)
        worker_id_list.append(agent_id + 1)

        # inference in every args.snap rounds
        logging.info("---------Test {} ------------".format(rnd))
        if rnd % args.snap == 0:
            if args.aggr != 'lockdown':
                val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(global_model, criterion, val_loader,
                                                                                    args, rnd, num_target)

                poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(global_model, criterion,
                                                                                poisoned_val_loader, args, rnd, num_classes=num_target)
                
                poison_loss, (poison_acc, _), fail_samples = utils.get_loss_n_accuracy(global_model, criterion,
                                                                                    poisoned_val_only_x_loader, args,
                                                                                    rnd, num_target)
            else:
                test_model = copy.deepcopy(global_model)

                # CF
                for name, param in test_model.named_parameters():
                    mask = 0
                    for id, agent in enumerate(agents):
                        mask += old_mask[id][name].to(args.device)
                    param.data = torch.where(mask.to(args.device) >= args.theta_ld, param,
                                             torch.zeros_like(param))
                    # logging.info(torch.sum(mask.to(args.device) >= args.theta) / torch.numel(mask))
                val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                      val_loader,
                                                                                      args, rnd, num_target)
                
                poison_loss, (asr, _), _ = utils.get_loss_n_accuracy(test_model, criterion,
                                                                            poisoned_val_loader,
                                                                            args, rnd, num_target)

                poison_loss, (poison_acc, _), fail_samples = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                       poisoned_val_only_x_loader, args,
                                                                                       rnd, num_target)
                
                del test_model
            logging.info('Main task accuracy:      %.2f%%' % (val_acc * 100))
            logging.info('Backdoor task accuracy:  %.2f%%' % (asr * 100))
            logging.info('Robustness accuracy:     %.2f%%' % (poison_acc * 100))

            if val_acc > best_acc:
                best_acc = val_acc
                best_asr = asr
                best_bcdr_acc = poison_acc

        logging.info("------------------------------".format(rnd))

    logging.info('Best results:')
    logging.info('Main task accuracy:      %.2f%%' % (best_acc * 100))
    logging.info('Backdoor task accuracy:  %.2f%%' % (best_asr * 100))
    logging.info('Robustness accuracy:     %.2f%%' % (best_bcdr_acc * 100))

    if len(aggregator.tpr_history) > 0:
        logging.info('Avg TPR:                 %.2f%%' % ((sum(aggregator.tpr_history) / len(aggregator.tpr_history)) * 100))
        logging.info('Avg FPR:                 %.2f%%' % ((sum(aggregator.fpr_history) / len(aggregator.fpr_history)) * 100))

    logging.info('Training has finished!')
