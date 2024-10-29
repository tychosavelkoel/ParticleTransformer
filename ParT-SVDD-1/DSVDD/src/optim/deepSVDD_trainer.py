from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from copy import copy

import logging
import time
import torch
import torch.optim as optim
import numpy as np
import tqdm

class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, Q, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, min_epochs= 20, max_epochs= 10, warm_up_n_epochs: int = 0,
                 epsilon: float = 1e-5, delta: float = 1e-3):
        super().__init__(optimizer_name, lr, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.warm_up_n_epochs = warm_up_n_epochs  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated
        self.epsilon = epsilon
        self.delta = delta
        self.loss_epoch_val_min = 1e10

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.val_scores = None
        self.loss = None
        self.loss_val = None
        self.loss_condition = None
        self.loss_condition_val = None
        self.final_cost_condition = None
        
        self.Q = torch.tensor(Q, device=self.device)
        self.Train_observers = defaultdict(list)
        self.Test_observers = defaultdict(list)

    def train(self,  net: BaseNet, train_loader = None, val_loader = None, data_config = None, 
              steps_per_epoch = 100, steps_per_epoch_val = 50):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')
 
        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net,  data_config = data_config)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        track_loss_train = []
        track_loss_val = []
        track_loss_condition_train = []
        track_loss_condition_val = []
        
        loss_epoch_train = 1e5
        loss_epoch_val = 1e5
        patience = 0
        loss_condition_passed_flag = False # flag to check if loss condition has been satisfied
        early_stop_flag = False # flag to check if loss condition validation increases
        warm_up_flag = False # flag to check if loss condition is small enough to set radius
        
        epoch = 0
        while ((epoch < self.min_epochs or loss_condition_passed_flag == False) and (epoch < self.max_epochs) and (early_stop_flag == False)):
            epoch += 1
            scheduler.step()

            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch_train_prev = copy(loss_epoch_train)
            loss_epoch_val_prev = copy(loss_epoch_val)
            loss_epoch_train = 0.0
            loss_epoch_val = 0.0
            n_batches_train = 0
            n_batches_val = 0
            epoch_start_time = time.time()
            
            score_train = []
            score_val = []
            Train_observers = defaultdict(list)

            with tqdm.tqdm(train_loader) as tq:
                for X, y, Z in tq:
                    inputs = [X[k].to(self.device) for k in data_config.input_names]
                    torch.save(net(*inputs), 'Output_Train.pt')
                    # Zero the network parameter gradients
                    optimizer.zero_grad()

                    # Update network parameters via backpropagation: forward + backward + optimize
                    outputs = net(*inputs).flatten(start_dim = 1)
                        
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)

                    score_train.append(dist.cpu().data.numpy().tolist())
                    for k, v in Z.items():
                        Train_observers[k].append(v)   
                    
                    if self.objective == 'soft-boundary':
                        scores = dist - self.R ** 2
                        loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                        
                    else:
                        loss = torch.mean(dist)

                    loss.backward()
                    optimizer.step()
            

                    # Update hypersphere radius R on mini-batch distances
                    if (self.objective == 'soft-boundary') and (warm_up_flag == True):
                        self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                    
                    loss_epoch_train += loss.item()
                    n_batches_train += 1
                    if steps_per_epoch is not None and n_batches_train >= steps_per_epoch:
                        break
                track_loss_train.append(loss_epoch_train/n_batches_train)
           
            #validation
            with torch.no_grad():
                with tqdm.tqdm(val_loader) as tq:
                    for X, y, _ in tq:
                        inputs = [X[k].to(self.device) for k in data_config.input_names]

                        # Update network parameters via backpropagation: forward + backward + optimize
                        outputs = net(*inputs).flatten(start_dim = 1)

                        dist = torch.sum((outputs - self.c) ** 2, dim=1)
                        
                        score_val.append(dist.cpu().data.numpy().tolist())
                            
                        
                        if self.objective == 'soft-boundary':
                            scores = dist - self.R ** 2
                            loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                            
                        else:
                            loss = torch.mean(dist)

                        loss_epoch_val += loss.item()
                        n_batches_val += 1
                        if steps_per_epoch_val is not None and n_batches_val >= steps_per_epoch_val:
                            break
                    track_loss_val.append(loss_epoch_val/n_batches_val)
            
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch, self.max_epochs, epoch_train_time, loss_epoch_train / n_batches_train))
            
            #Determine loss condition
            loss_condition_train = (loss_epoch_train/n_batches_train - loss_epoch_train_prev/n_batches_train) ** 2
            track_loss_condition_train.append(loss_condition_train)
            print("loss_condition_train=", loss_condition_train)
            loss_condition_val = (loss_epoch_val/n_batches_val - loss_epoch_val_prev/n_batches_val) ** 2
            track_loss_condition_val.append(loss_condition_val)


            
            #check if radius can be set
            if loss_condition_train <= self.delta:
                warm_up_flag = True
                print("radius set")

            #check if loss condition is satisfied after radius is set
            if self.R > 0:
                if loss_condition_train <= self.epsilon:
                    loss_condition_passed_flag = True
                else:
                    loss_condition_passed_flag = False
            
            #early stopping mechanism
            if epoch >= self.min_epochs:
                if loss_epoch_val < self.loss_epoch_val_min:
                    self.loss_epoch_val_min = loss_epoch_val
                    patience = 0
                elif loss_epoch_val/n_batches_val - self.loss_epoch_val_min/n_batches_val > 0.1:
                    patience += 1
                    if patience >= 5:
                        early_stop_flag = True
       
        if loss_condition_train > self.epsilon:
            if epoch < self.max_epochs:
                logger.info("Algorithm failed: early stop at epoch {}".format(epoch))
            else:
                logger.info("Algorithm failed: not done learning in max = {} epochs".format(epoch))
        else:
            logger.info("Model done learning in {} epochs".format(epoch))

        logger.info("Final loss condition training = {}".format(loss_condition_train)) 
        logger.info("Final loss condition validation = {}".format(loss_condition_val))  


        #save results
        score_train = np.concatenate(score_train)
        score_val = np.concatenate(score_val)
        self.Train_observers = Train_observers

        self.train_time = time.time() - start_time
        self.train_scores = list(zip(score_train))
        self.val_scores = list(zip(score_val))
        self.loss = track_loss_train
        self.loss_val = track_loss_val
        self.loss_condition = track_loss_condition_train
        self.loss_condition_val = track_loss_condition_val
        self.final_loss_condition = loss_condition_train
 
        print("train:", score_train.shape)
        
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, net: BaseNet, test_loader = None, data_config = None, steps_per_epoch = 10):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        score = []
        net.eval()
        n_batches = 0

        with torch.no_grad():
            with tqdm.tqdm(test_loader) as tq:
                for X, y, Z in tq:
                    inputs = [X[k].to(self.device) for k in data_config.input_names]
                    torch.save(net(*inputs), 'Output_Test.pt')
                    outputs = net(*inputs).flatten(start_dim = 1)
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)

                    if self.objective == 'soft-boundary':
                        scores = dist - self.R ** 2
                    else:
                        scores = dist

                    # Save triples of (idx, label, score) in a list --> save score in a list
                    score.append(dist.cpu().data.numpy().tolist())
                    
                    for k, v in Z.items():
                        self.Test_observers[k].append(v)

                    self.Q.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                    n_batches+=1
                    if steps_per_epoch is not None and n_batches >= steps_per_epoch:
                        break

        score = np.concatenate(score)
        
        
        print("test:", score.shape)


        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = list(score)

        # Compute AUC
        #_, labels, scores = zip(*idx_label_score)
        #labels = np.array(labels)
        #scores = np.array(scores)

        #self.test_auc = roc_auc_score(labels, scores)
        #logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1,  data_config = None,):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(1*128, device=self.device) #torch.zeros(net.rep_dim, device=self.device)
        
        num_batches = 0
        with torch.no_grad():
            with tqdm.tqdm(train_loader) as tq:
                for X, y, Z in tq:
                    inputs = [X[k].to(self.device) for k in data_config.input_names]
                    outputs = net(*inputs).flatten(start_dim = 1)
                    n_samples += outputs.shape[0]
                    c += torch.sum(outputs, dim=0)
                    num_batches += 1

                    if num_batches >= 30:
                        break
        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
