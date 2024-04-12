from flgo.algorithm.fedbase import BasicServer, BasicClient
import torch
from flgo.utils import fmodule
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
import os
import flgo
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp
import flgo.experiment.analyzer


class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'mu':0.01})

class Client(BasicClient):
    def get_batch_data(self, data_loader):
        """
        Get the batch of training data
        Returns:
            a batch of data
        """
        if self._train_loader is None:
            self._train_loader = data_loader
        try:
            batch_data = next(data_loader)
        except Exception as e:
            data_loader = iter(self._train_loader)
            batch_data = next(data_loader)
        # clear local_movielens_recommendation DataLoader when finishing local_movielens_recommendation training
        self.current_steps = (self.current_steps + 1) % self.num_steps
        if self.current_steps == 0:
            self.data_loader = None
            self._train_loader = None
        return batch_data
    # @fmodule.with_multi_gpus
    def train(self, model):
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        data_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.loader_num_workers,
            pin_memory=self.option['pin_memory'],
            drop_last=not self.option.get('no_drop_last', False)
        )
        MAX_GRAD_NORM = 1.2
        EPSILON = 50.0
        DELTA = 1e-5
        EPOCHS = 5
        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            # noise_multiplier=1.0,
            # max_grad_norm=1.0,
            epochs=EPOCHS,
            target_epsilon=EPSILON,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )


        # for iter in range(self.num_steps):
        #     # get a batch of data
        #     batch_data = data_loader
        #     model.zero_grad()
        #     # calculate the loss of the model on batched dataset through task-specified calculator
        #     loss = self.calculator.compute_loss(model, batch_data)['loss']
        #     loss.backward()
        #     if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
        #     optimizer.step()
        # return
        # for iter in range(self.num_steps):
        #     for batch_data in data_loader:
        #         inputs, labels = batch_data
        #         model.zero_grad()
        #         optimizer.zero_grad()
        #         # 计算模型在批数据上的损失
        #         loss = self.calculator.compute_loss(model, (inputs, labels))['loss']
        #         loss.backward()
        #         if self.clip_grad > 0:
        #             torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
        #         optimizer.step()
        # return
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data(data_loader)
            model.zero_grad()
            optimizer.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
        return
class fedavg:
    Server = Server
    Client = Client



task = './my_task1'
flgo.gen_task_by_(mnist, fbp.IIDPartitioner(num_clients=10), task)
runner = flgo.init(task, fedavg, option={'num_rounds':5, 'local_test':True, 'learning_rate':0.005, })
runner.run()
analysis_plan = {
    'Selector':{
        'task': task,
        'header':['fedavg']
    },
    'Painter':{
        'Curve':[
            {'args':{'x': 'communication_round', 'y':'val_loss'}, 'fig_option':{'title':'valid loss on MNIST'}},
            {'args':{'x': 'communication_round', 'y':'val_accuracy'},  'fig_option':{'title':'valid accuracy on MNIST'}},
        ]
    }
}
flgo.experiment.analyzer.show(analysis_plan)