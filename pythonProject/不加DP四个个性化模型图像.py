import flgo.experiment.analyzer
task='./my_task1'
analysis_plan = {
    'Selector':{
        'task': task,
        'header':['fedpac', 'fedala','fedrod','pfedme']
    },
    'Painter':{
        'Curve':[
            {'args':{'x': 'communication_round', 'y':'val_loss'}, 'fig_option':{'title':'valid loss on Synthetic'}},
            {'args':{'x': 'communication_round', 'y':'val_accuracy'},  'fig_option':{'title':'valid accuracy on Synthetic'}},
        ]
    }
}
flgo.experiment.analyzer.show(analysis_plan)