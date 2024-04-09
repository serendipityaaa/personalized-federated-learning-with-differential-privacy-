import flgo
import os
import flgo.benchmark.mnist_classification.model.cnn as cnn
import flgo.benchmark.mnist_classification.model.mlp as mlp
import flgo.algorithm.fedavg as fedavg

# 设置联邦任务，如果任务不存在，取消注释下面两行代码生成该任务
task = './test_mnist'
config = {'benchmark':{'name':'flgo.benchmark.mnist_classification'},'partitioner':{'name': 'IIDPartitioner','para':{'num_clients':100}}}
if not os.path.exists(task): flgo.gen_task(config, task_path = task)

cnn_runner = flgo.init(task, fedavg, option={'num_rounds':5, 'num_epochs':1}, model=cnn)
mlp_runner = flgo.init(task, fedavg, option={'num_rounds':5, 'num_epochs':1}, model=mlp)
cnn_runner.run()
mlp_runner.run()
import flgo.experiment.analyzer
analysis_plan = {
    'Selector':{
        'task': task,
        'header':['fedavg'],
        'filter':{'M':['cnn', 'mlp']},
        'legend_with':['M']
    },
    'Painter':{
        'Curve':[
            {'args':{'x': 'communication_round', 'y':'val_loss'}, 'fig_option':{'title':'valid loss on MNIST'}},
            {'args':{'x': 'communication_round', 'y':'val_accuracy'},  'fig_option':{'title':'valid accuracy on MNIST'}},
        ]
    }
}
flgo.experiment.analyzer.show(analysis_plan)