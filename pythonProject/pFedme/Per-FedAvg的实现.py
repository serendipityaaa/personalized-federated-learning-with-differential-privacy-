from pFedme import pfedme
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp
from flgo.experiment.logger.pool import PFLLogger
import flgo.experiment.analyzer
task = './my_task1'
flgo.gen_task_by_(mnist, fbp.IIDPartitioner(num_clients=4), task)
runner = flgo.init(task, pfedme, option={'num_rounds':20, 'local_test':True, 'learning_rate':0.005, }, Logger=PFLLogger)
runner.run()
analysis_plan = {
    'Selector':{
        'task': task,
        'header':['pfedme']
    },
    'Painter':{
        'Curve':[
            {'args':{'x': 'communication_round', 'y':'val_loss'}, 'fig_option':{'title':'valid loss on MNIST'}},
            {'args':{'x': 'communication_round', 'y':'val_accuracy'},  'fig_option':{'title':'valid accuracy on MNIST'}},
        ]
    }
}
flgo.experiment.analyzer.show(analysis_plan)