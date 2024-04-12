import flgo.algorithm.fedbase as fedbase
import flgo.algorithm.fedavg as fedavg
import flgo.utils.fmodule as fmodule
import os
import flgo
import flgo.experiment.analyzer
import copy
import torch
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp

task = './my_task1'
# flgo.gen_task_by_(mnist, fbp.IIDPartitioner(num_clients=10), task)
# runner = flgo.init(task, fedavg, option={'num_rounds':5, 'local_test':True, 'learning_rate':0.005, })
# runner.run()
analysis_plan = {
    'Selector':{
        'task': task,
        'header':['fedavg','fedavg_DP']
    },
    'Painter':{
        'Curve':[
            {'args':{'x': 'communication_round', 'y':'val_loss'}, 'fig_option':{'title':'valid loss on MNIST'}},
            {'args':{'x': 'communication_round', 'y':'val_accuracy'},  'fig_option':{'title':'valid accuracy on MNIST'}},
        ]
    }
}
flgo.experiment.analyzer.show(analysis_plan)