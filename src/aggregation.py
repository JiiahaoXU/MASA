import copy

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
import numpy as np
import logging
from utils import vector_to_model, vector_to_name_param, vector_to_model_wo_load
import sklearn.metrics.pairwise as smp
from geom_median.torch import compute_geometric_median 
from torch.utils.data import DataLoader

class Aggregation():
    def __init__(self, agent_data_sizes, n_params, poisoned_val_loader, args, writer, server_dataset):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.writer = writer
        self.server_lr = args.server_lr
        self.n_params = n_params
        self.poisoned_val_loader = poisoned_val_loader
        self.cum_net_mov = 0
        self.update_last_round = None
        self.memory_dict = dict()
        self.wv_history = []
        self.global_model = None
        
        self.tpr_history = []
        self.fpr_history = []
        self.defense_data_loader = self.get_train_loader(server_dataset)
         
    def aggregate_updates(self, global_model, agent_updates_dict):
        self.global_model = global_model
        

        if self.args.attack == 'lie':
            all_updates = []
            for id, update in agent_updates_dict.items():
                all_updates.append(agent_updates_dict[id])

            est_updates = torch.stack(all_updates)

            mu = torch.mean(est_updates, dim=0)
            sigma = torch.std(est_updates, dim=0)
            z = 1.5  
            minn = mu - z * sigma
            maxx = mu + z * sigma
            for id, update in agent_updates_dict.items():
                if id < self.args.num_corrupt:
                    # agent_updates_dict[id] *= total_num_dps_per_round / num_dps_poisoned_dataset
                    agent_updates_dict[id] = torch.where(agent_updates_dict[id] < minn, minn, agent_updates_dict[id])
                    agent_updates_dict[id] = torch.where(agent_updates_dict[id] > maxx, maxx, agent_updates_dict[id])

        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)
        if self.args.aggr != "rlr":
            lr_vector = lr_vector
        else:
            lr_vector, _ = self.compute_robustLR(agent_updates_dict)
        aggregated_updates = 0
        cur_global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.args.aggr=='avg' or self.args.aggr == 'rlr' or self.args.aggr == 'lockdown':          
            aggregated_updates = self.agg_avg(agent_updates_dict)
        elif self.args.aggr == 'masa':
            aggregated_updates = self.agg_masa(agent_updates_dict, cur_global_params)
        elif self.args.aggr == 'mul_metric':
            aggregated_updates = self.agg_mul_metric(agent_updates_dict)
        elif self.args.aggr == 'fg':
            aggregated_updates = self.agg_foolsgold(agent_updates_dict)
        elif self.args.aggr == "mkrum":
            aggregated_updates = self.agg_mkrum(agent_updates_dict)
        elif self.args.aggr == "rfa":
            aggregated_updates = self.agg_rfa(agent_updates_dict)
        elif self.args.aggr == "bulyan":
            aggregated_updates = self.agg_bulyan(agent_updates_dict)
        neurotoxin_mask = {}
        updates_dict = vector_to_name_param(aggregated_updates, copy.deepcopy(global_model.state_dict()))
        for name in updates_dict:
            updates = updates_dict[name].abs().view(-1)
            gradients_length = torch.numel(updates)
            _, indices = torch.topk(-1 * updates, int(gradients_length * self.args.dense_ratio))
            mask_flat = torch.zeros(gradients_length)
            mask_flat[indices.cpu()] = 1
            neurotoxin_mask[name] = (mask_flat.reshape(updates_dict[name].size()))

        cur_global_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float()
        vector_to_model(new_global_params, global_model)
        return updates_dict, neurotoxin_mask

    def get_train_loader(self, server_dataset):
        print('==> Preparing server data..')
        train_loader = DataLoader(server_dataset, batch_size=64, shuffle=True)

        return train_loader
    def agg_masa(self, agent_updates_dict, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id


        stacked_updates = torch.stack(local_updates, dim=0)
        temp_model = copy.deepcopy(self.global_model)
        unlearning_accumulated_loss = []
        pre_aggregation = self.agg_avg(agent_updates_dict)


        ###################################
        ## Perform Individual Unlearning ##
        ###################################
        
        for _id, update in zip(range(len(stacked_updates)), stacked_updates):
            
            # Pre-unlearning model fusion
            local_model_para = (flat_global_model + update * self.args.fusion_lambda + pre_aggregation * (1 - self.args.fusion_lambda)).float()

            fused_model_dict = vector_to_model_wo_load(local_model_para, temp_model)
            temp_model.load_state_dict(fused_model_dict)

            # Unlearning
            criterion = torch.nn.CrossEntropyLoss().cuda()
            unlearning_optimizer = torch.optim.SGD(temp_model.parameters(), lr=self.args.unlearn_lr, momentum=self.args.unlearn_moment)

            cum_unlearning_loss = 0
            for epoch in range(1, 5 + 1):
                unlearning_loss, unlearning_acc, temp_model = self.train_step_unlearning(model=temp_model, criterion=criterion, optimizer=unlearning_optimizer,
                                        data_loader=self.defense_data_loader)
                
                print('Client: %d \t unlearning epoch: %d \t unlearning loss: %.4f \t unlearning acc: %.4f' % (_id, epoch, unlearning_loss, unlearning_acc))

                cum_unlearning_loss += unlearning_loss
            print('--------')
            unlearning_accumulated_loss.append(cum_unlearning_loss)



        full_set = set([i for i in range(len(unlearning_accumulated_loss))])

        print('######### Accumulated unlearning loss #########')
        print([round(item, 4) for item in unlearning_accumulated_loss])

        loss_std = np.std(unlearning_accumulated_loss)
        loss_med = np.median(unlearning_accumulated_loss)
        mds = []
        for i in range(len(unlearning_accumulated_loss)):
            mds.append((unlearning_accumulated_loss[i] - loss_med) / loss_std)
        
        print('######### Median Deviation Score #########')
        print([round(item, 4) for item in mds])

        # Determine benign ones with filter_delta
        benign_set = full_set.intersection(set([int(i) for i in np.argwhere(np.array(mds) < self.args.filter_delta)]))

        # Calculate TPR and FPR
        correct = 0
        for idx in benign_set:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_set:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_set))

        logging.info('FPR:       %.2f%%'  % (FPR * 100))
        logging.info('TPR:       %.2f%%' % (TPR * 100))

        self.tpr_history.append(TPR)
        self.fpr_history.append(FPR)
        
        # Aggregate benign local model updates
        benign_dict = {}
        for idx in benign_set:
            benign_dict[chosen_clients[idx]] = stacked_updates[idx]

        return self.agg_avg(benign_dict)
    
    def train_step_unlearning(self, model, criterion, optimizer, data_loader):
        model.train()
        total_correct = 0
        total_loss = 0.0
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = model(images)

            loss = criterion(output, labels)

            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()
            (-loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc, model
    
    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """

        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data
        return  sm_updates / total_data

    def compute_robustLR(self, agent_updates_dict):

        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask=torch.zeros_like(sm_of_signs)
        mask[sm_of_signs < self.args.theta] = 0
        mask[sm_of_signs >= self.args.theta] = 1
        sm_of_signs[sm_of_signs < self.args.theta] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
        return sm_of_signs.to(self.args.device), mask
    
    def agg_rfa(self, agent_updates_dict):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        n = len(local_updates)
        grads = torch.stack(local_updates, dim=0)
        weights = torch.ones(n).to(self.args.device)  
        gw = compute_geometric_median(local_updates, weights).median
        for i in range(2):
            weights = torch.mul(weights, torch.exp(-1.0*torch.norm(grads-gw, dim=1)))
            gw = compute_geometric_median(local_updates, weights).median

        aggregated_model = gw
        return aggregated_model
    
    def agg_mkrum(self, agent_updates_dict):
        krum_param_m = 10
        def _compute_krum_score( vec_grad_list, byzantine_client_num):
            krum_scores = []
            num_client = len(vec_grad_list)
            for i in range(0, num_client):
                dists = []
                for j in range(0, num_client):
                    if i != j:
                        dists.append(
                            torch.norm(vec_grad_list[i]- vec_grad_list[j])
                            .item() ** 2
                        )
                dists.sort()  # ascending
                score = dists[0: num_client - byzantine_client_num - 2]
                krum_scores.append(sum(score))
            return krum_scores

        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            # local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        # Compute list of scores
        __nbworkers = len(agent_updates_dict)
        krum_scores = _compute_krum_score(agent_updates_dict, self.args.num_corrupt)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: krum_param_m]

        print('%d clients are selected' % len(score_index))
        return_gradient = [agent_updates_dict[i] for i in score_index]

        return sum(return_gradient)/len(return_gradient)
    
    def agg_mul_metric(self, agent_updates_dict):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

        vectorize_nets = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]

        cos_dis = [0.0] * len(vectorize_nets)
        length_dis = [0.0] * len(vectorize_nets)
        manhattan_dis = [0.0] * len(vectorize_nets)
        for i, g_i in enumerate(vectorize_nets):
            for j in range(len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]

                    cosine_distance = float(
                        (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)   #Compute the different value of cosine distance
                    manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))    #Compute the different value of Manhattan distance
                    length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))    #Compute the different value of Euclidean distance

                    cos_dis[i] += cosine_distance
                    length_dis[i] += length_distance
                    manhattan_dis[i] += manhattan_distance

        tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T

        cov_matrix = np.cov(tri_distance.T)
        inv_matrix = np.linalg.inv(cov_matrix)

        ma_distances = []
        for i, g_i in enumerate(vectorize_nets):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        print(scores)

        p = 0.3
        p_num = p*len(scores)
        topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]   #sort

        print(topk_ind)
        current_dict = {}

        for idx in topk_ind:
            current_dict[chosen_clients[idx]] = agent_updates_dict[chosen_clients[idx]]

        # return self.agg_avg_norm_clip(current_dict)
        update = self.agg_avg(current_dict)

        return update

    def agg_foolsgold(self, agent_updates_dict):
        def foolsgold(grads):
            """
            :param grads:
            :return: compute similatiry and return weightings
            """
            n_clients = grads.shape[0]
            cs = smp.cosine_similarity(grads) - np.eye(n_clients)

            maxcs = np.max(cs, axis=1)
            # pardoning
            for i in range(n_clients):
                for j in range(n_clients):
                    if i == j:
                        continue
                    if maxcs[i] < maxcs[j]:
                        cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
            wv = 1 - (np.max(cs, axis=1))

            wv[wv > 1] = 1
            wv[wv < 0] = 0

            alpha = np.max(cs, axis=1)

            # Rescale so that max value is wv
            wv = wv / np.max(wv)
            wv[(wv == 1)] = .99

            # Logit function
            wv = (np.log(wv / (1 - wv)) + 0.5)
            wv[(np.isinf(wv) + wv > 1)] = 1
            wv[(wv < 0)] = 0

            # wv is the weight
            return wv, alpha

        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        names = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

        client_grads = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]
        grad_len = np.array(client_grads[0].shape).prod()
        # print("client_grads size", client_models[0].parameters())
        # grad_len = len(client_grads)
        # if self.memory is None:
        #     self.memory = np.zeros((self.num_clients, grad_len))
        if len(names) < len(client_grads):
            names = np.append([-1], names)  # put in adv

        num_clients = num_chosen_clients
        memory = np.zeros((num_clients, grad_len))
        grads = np.zeros((num_clients, grad_len))

        for i in range(len(client_grads)):
            # grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
            grads[i] = np.reshape(client_grads[i], (grad_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]] += grads[i]
            else:
                self.memory_dict[names[i]] = copy.deepcopy(grads[i])
            memory[i] = self.memory_dict[names[i]]
        # self.memory += grads
        use_memory = False

        if use_memory:
            wv, alpha = foolsgold(None)  # Use FG
        else:
            wv, alpha = foolsgold(grads)  # Use FG
        self.wv_history.append(wv)

        print(len(client_grads), len(wv))

        
        weighted_updates = [update * wv[i] for update, i in zip(agent_updates_dict.values(), range(len(wv)))]

        aggregated_model = torch.mean(torch.stack(weighted_updates, dim=0), dim=0)

        print(aggregated_model.shape)

        return aggregated_model

