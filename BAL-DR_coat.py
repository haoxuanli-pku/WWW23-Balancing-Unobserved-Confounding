import os
import numpy as np
import random

import torch
import torch.nn as nn

from model import *

import arguments

import utils.load_dataset
import utils.data_loader
import utils.metrics
from utils.early_stop import EarlyStopping, Stop_args

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def para(args): 
    if args.dataset == 'coat':
        args.training_args = {'batch_size': 128, 'epochs': 500, 'patience': 60, 'block_batch': [64, 64]}
        args.base_model_args = {'emb_dim': 10, 'learning_rate': 1e-3, 'weight_decay': 1}
        args.weight1_model_args = {'emb_dim': 10, 'learning_rate': 1e-1, 'weight_decay': 1}
        args.weight2_model_args = {'emb_dim': 10, 'learning_rate': 1e-1, 'weight_decay': 1}
        args.imputation_model_args = {'emb_dim': 2, 'learning_rate': 1e-3, 'weight_decay': 1}
    else:
        print('invalid arguments')
        os._exit()
                

def train_and_eval(train_data, unif_train_data, val_data, test_data, device = 'cuda', 
        model_args: dict = {'emb_dim': 64, 'learning_rate': 0.01, 'weight_decay': 0.1}, 
        weight1_model_args: dict = {'emb_dim': 4, 'learning_rate': 0.1, 'weight_decay': 0.1}, 
        weight2_model_args: dict = {'emb_dim': 4, 'learning_rate': 0.1, 'weight_decay': 0.1}, 
        imputation_model_args: dict = {'emb_dim': 2, 'learning_rate': 0.1, 'weight_decay': 0.1},
        training_args: dict =  {'batch_size': 1024, 'epochs': 100, 'patience': 20, 'block_batch': [1000, 100]}): 

    # build data_loader. 
    train_loader = utils.data_loader.Block(train_data, u_batch_size=training_args['block_batch'][0], i_batch_size=training_args['block_batch'][1], device=device)
    val_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(val_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)
    test_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(test_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)

    # uniform data
    users_unif = unif_train_data._indices()[0]
    items_unif = unif_train_data._indices()[1]
    y_unif = unif_train_data._values()

    # Naive Bayes propensity Estimator
    def Naive_Bayes_Propensity(train, unif): 
        # the implementation of naive bayes propensity
        P_Oeq1 = train._nnz() / (train.size()[0] * train.size()[1])

        y_unique = torch.unique(train._values())
        P_y_givenO = torch.zeros(y_unique.shape).to(device)
        P_y = torch.zeros(y_unique.shape).to(device)

        for i in range(len(y_unique)): 
            P_y_givenO[i] = torch.sum(train._values() == y_unique[i]) / torch.sum(torch.ones(train._values().shape).to(device))
            P_y[i] = torch.sum(unif._values() == y_unique[i]) / torch.sum(torch.ones(unif._values().shape).to(device))

        Propensity = P_y_givenO * P_Oeq1 / P_y

        return y_unique, Propensity

    y_unique, Propensity = Naive_Bayes_Propensity(train_data, unif_train_data)
    InvP = torch.reciprocal(Propensity)
    
    # data shape
    n_user, n_item = train_data.shape

    # loss_criterion
    none_criterion = nn.MSELoss(reduction='none')
    mean_criterion = nn.MSELoss(reduction='mean')
    sum_criterion = nn.MSELoss(reduction='sum')
    l1_criterion = nn.L1Loss()

    # base model and its optimizer. 
    model = MF(n_user, n_item, dim=model_args['emb_dim'], dropout=0).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=model_args['learning_rate'], weight_decay=0)

    weight1_model = MF(n_user, n_item, dim= weight1_model_args['emb_dim'], dropout=0).to(device)
    weight1_optimizer = torch.optim.SGD(weight1_model.parameters(), lr=weight1_model_args['learning_rate'], weight_decay=0)

    weight2_model = MF(n_user, n_item, dim= weight2_model_args['emb_dim'], dropout=0).to(device)
    weight2_optimizer = torch.optim.SGD(weight2_model.parameters(), lr=weight2_model_args['learning_rate'], weight_decay=0) 

    imputation_model = MF(n_user, n_item, dim=imputation_model_args['emb_dim'], dropout=0).to(device)
    imputation_optimizer = torch.optim.SGD(imputation_model.parameters(), lr=imputation_model_args['learning_rate'], weight_decay=0)

    # begin training
    stopping_args = Stop_args(patience=training_args['patience'], max_epochs=training_args['epochs'])
    early_stopping = EarlyStopping(model, **stopping_args)
    for epo in range(early_stopping.max_epochs):
        training_loss = 0
        for u_batch_idx, users in enumerate(train_loader.User_loader): 
            for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                users_train, items_train, y_train = train_loader.get_batch(users, items)

                inverse_p = torch.ones(y_train.shape).to(device)
                for i in range(len(y_unique)): 
                    inverse_p[y_train == y_unique[i]] = InvP[i]

                # step 1: update imptation error model
                imputation_model.train()

                e_hat = imputation_model(users_unif, items_unif) # imputation error
                e = y_train - model(users_unif, items_unif) # prediction error
                loss_imp = sum_criterion(e_hat, e) + imputation_model_args['weight_decay'] * imputation_model.l2_norm(users_train, items_train)

                imputation_optimizer.zero_grad()
                loss_imp.backward(retain_graph=True)
                imputation_optimizer.step()

                all_pair = torch.cartesian_prod(users, items)
                users_all, items_all = all_pair[:,0], all_pair[:,1]

                #step2: update weight1_model
                weight1_model.train()
                w1 = weight1_model(users_all, items_all) # compute weight1
                w1 = torch.softmax(w1, dim=0) 
         
                 # compute loss_all
                y_hat_all = model(users_all, items_all)
                y_hat_all_detach = torch.detach(y_hat_all)
                g_all = imputation_model(users_all, items_all)
                g_all_detach = torch.detach(g_all)
                loss_all = none_criterion(y_hat_all_detach, g_all_detach+ y_hat_all_detach) # sum(e_hat)
                loss_all1 = torch.sum(w1*loss_all)

                w2 = weight2_model(users_train, items_train) # compute weight
                w2 = torch.softmax(w2, dim=0)
                w2_detach = torch.detach(w2)
                #compute loss_obs
                y_hat_obs = model(users_train, items_train)
                y_hat_obs_detach = torch.detach(y_hat_obs)
                g_obs = imputation_model(users_train, items_train)
                g_obs_detach = torch.detach(g_obs)
                e_obs = none_criterion(y_hat_obs_detach, y_train)
                e_hat_obs = none_criterion(y_hat_obs_detach, g_obs_detach + y_hat_obs_detach)
                cost_obs = e_obs - e_hat_obs
                loss_obs1 = torch.sum(inverse_p * w2_detach * cost_obs)    
                mnar_loss1 = loss_all1 + loss_obs1

                y_hat_unif = model(users_unif, items_unif)
                y_hat_unif_detach = torch.detach(y_hat_unif)
                e_unif = none_criterion(y_hat_unif_detach, y_unif)
                mar_loss = (1/len(y_unif)) * torch.sum(e_unif)
                loss_w1 = 1e-1 * sum_criterion(mnar_loss1,mar_loss) + 10 * torch.dot(w1,torch.log(w1)) + weight1_model_args['weight_decay'] * weight1_model.l2_norm(users_train, items_train)           

                weight1_optimizer.zero_grad()
                loss_w1.backward(retain_graph=True)
                weight1_optimizer.step()          

                #step3: update weight2 model
                weight2_model.train()
                w2 = weight2_model(users_train, items_train) # compute weight
                w2 = torch.softmax(w2, dim=0)

                w1 = weight1_model(users_all, items_all) # compute weight1
                w1 = torch.softmax(w1, dim=0)
                w1_detach = torch.detach(w1) 

                y_hat_all = model(users_all, items_all)
                y_hat_all_detach = torch.detach(y_hat_all)
                g_all = imputation_model(users_all, items_all)
                g_all_detach = torch.detach(g_all)
                loss_all = none_criterion(y_hat_all_detach, g_all_detach+ y_hat_all_detach) # sum(e_hat)
                loss_all2 = torch.sum(w1_detach * loss_all)

                y_hat_obs = model(users_train, items_train)
                y_hat_obs_detach = torch.detach(y_hat_obs)
                g_obs = imputation_model(users_train, items_train)
                g_obs_detach = torch.detach(g_obs)
                e_obs = none_criterion(y_hat_obs_detach, y_train)
                e_hat_obs = none_criterion(y_hat_obs_detach, g_obs_detach + y_hat_obs_detach)
                cost_obs = e_obs - e_hat_obs
                loss_obs2 = torch.sum(inverse_p * w2 * cost_obs)    
                mnar_loss2 = loss_all2 + loss_obs2 
                loss_w2 = 1e-1 * sum_criterion(mnar_loss2,mar_loss) + 10 * torch.dot(w2,torch.log(w2)) + weight2_model_args['weight_decay'] * weight2_model.l2_norm(users_train, items_train)           
                              
                weight2_optimizer.zero_grad()
                loss_w2.backward(retain_graph=True)
                weight2_optimizer.step()

                # step4: update prediction model
                model.train()
                w1 = weight1_model(users_all, items_all) # compute weight1
                w1 = torch.softmax(w1, dim=0) 
                w1_detach = torch.detach(w1)      

                # compute loss_all
                y_hat_all = model(users_all, items_all)
                y_hat_all_detach = torch.detach(y_hat_all)
                g_all = imputation_model(users_all, items_all)
                loss_all = none_criterion(y_hat_all, g_all + y_hat_all_detach) # sum(e_hat)
                loss_all = torch.sum(w1_detach*loss_all)

                w2 = weight2_model(users_train, items_train) # compute weight
                w2 = torch.softmax(w2, dim=0)
                w2_detach = torch.detach(w2)
                #compute loss_obs
                y_hat_obs = model(users_train, items_train)
                y_hat_obs_detach = torch.detach(y_hat_obs)
                g_obs = imputation_model(users_train, items_train)
                e_obs = none_criterion(y_hat_obs, y_train)
                e_hat_obs = none_criterion(y_hat_obs, g_obs + y_hat_obs_detach)
                cost_obs = e_obs - e_hat_obs
                loss_obs = torch.sum(inverse_p * w2_detach * cost_obs) 
                loss = loss_all + loss_obs + model_args['weight_decay'] * model.l2_norm(users_all, items_all)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()               
        

        model.eval()
        with torch.no_grad():
            # train metrics
            train_pre_ratings = torch.empty(0).to(device)
            train_ratings = torch.empty(0).to(device)
            for u_batch_idx, users in enumerate(train_loader.User_loader): 
                for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                    users_train, items_train, y_train = train_loader.get_batch(users, items)
                    pre_ratings = model(users_train, items_train)
                    train_pre_ratings = torch.cat((train_pre_ratings, pre_ratings))
                    train_ratings = torch.cat((train_ratings, y_train))

            # validation metrics
            val_pre_ratings = torch.empty(0).to(device)
            val_ratings = torch.empty(0).to(device)
            for batch_idx, (users, items, ratings) in enumerate(val_loader):
                pre_ratings = model(users, items)
                val_pre_ratings = torch.cat((val_pre_ratings, pre_ratings))
                val_ratings = torch.cat((val_ratings, ratings))
            
        train_results = utils.metrics.evaluate(train_pre_ratings, train_ratings, ['MSE', 'NLL'])
        val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'NLL', 'AUC'])

        print('Epoch: {0:2d} / {1}, Traning: {2}, Validation: {3}'.
                format(epo, training_args['epochs'], ' '.join([key+':'+'%.3f'%train_results[key] for key in train_results]), 
                ' '.join([key+':'+'%.3f'%val_results[key] for key in val_results])))

        
        if early_stopping.check([val_results['AUC']], epo):
            break

    # testing loss
    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    model.load_state_dict(early_stopping.best_state)

    # validation metrics
    val_pre_ratings = torch.empty(0).to(device)
    val_ratings = torch.empty(0).to(device)
    for batch_idx, (users, items, ratings) in enumerate(val_loader):
        pre_ratings = model(users, items)
        val_pre_ratings = torch.cat((val_pre_ratings, pre_ratings))
        val_ratings = torch.cat((val_ratings, ratings))

    # test metrics
    test_users = torch.empty(0, dtype=torch.int64).to(device)
    test_items = torch.empty(0, dtype=torch.int64).to(device)
    test_pre_ratings = torch.empty(0).to(device)
    test_ratings = torch.empty(0).to(device)
    for batch_idx, (users, items, ratings) in enumerate(test_loader):
        pre_ratings = model(users, items)
        test_users = torch.cat((test_users, users))
        test_items = torch.cat((test_items, items))
        test_pre_ratings = torch.cat((test_pre_ratings, pre_ratings))
        test_ratings = torch.cat((test_ratings, ratings))

    val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'NLL', 'AUC'])
    test_results = utils.metrics.evaluate(test_pre_ratings, test_ratings, ['MSE', 'NLL', 'AUC', 'Recall_Precision_NDCG@'], users=test_users, items=test_items)
    print('-'*30)
    print('The performance of validation set: {}'.format(' '.join([key+':'+'%.3f'%val_results[key] for key in val_results])))
    print('The performance of testing set: {}'.format(' '.join([key+':'+'%.3f'%test_results[key] for key in test_results])))
    print('-'*30)
    return val_results,test_results

if __name__ == "__main__": 
    args = arguments.parse_args()
    para(args)
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train, unif_train, validation, test = utils.load_dataset.load_dataset(data_name=args.dataset, type = 'explicit', seed = args.seed, device=device)
    train_and_eval(train, unif_train, validation, test, device, model_args = args.base_model_args, training_args = args.training_args,weight1_model_args = args.weight1_model_args, weight2_model_args = args.weight2_model_args)
