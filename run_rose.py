import os, sys

import radical.pilot as rp

from rose.learner import ActiveLearner
from rose.engine import Task, ResourceEngine

engine = ResourceEngine({'resource': 'nersc.perlmutter_gpu', 
                         'runtime' : 60, 
                         'access_schema':'interactive',
                         'project' : "m2616_g",
                         'cores'   : 64, 
                         'gpus'    : 4})

learner = ActiveLearner(engine=engine)
code_path = f'{sys.executable} {os.getcwd()}'


@learner.training_task
def training(*args, <extra_params>):
    return Task(executable=f'{code_path}/train.py <extra_params>')

@learner.active_learn_task
def active_learning(*args, <extra_params>):
    return Task(executable=f'{code_path}/active.py <extra_params>')

# This is to build a sub-workflow that includes training/active learning loop for a specific AL algorithm
def teach_single_pipeline(<some_params>):
    iter_id = 0
    train = training(<some_params>)
    # Wait for train to finish
    train.result()
    while iter_id < num_iter - 1:   # n iter means in total n traning and n-1 al
        active_learn = active_learning(<some_params>)
        # Wait for al to finish
        active_learn.result()
        train = training(sim, <some_params>)
        # Wait for train to finish
        train.result()
        iter_id += 1

def teach():
    submitted_pipelines = []
    # Some conf list we set, like seed
    conf_list = []
    async_pipeline = learner.as_async(teach_single_pipeline)
    for model_type in ["BNN", "GPR"]:
        for conf in conf_list:
            submitted_pipelines.append(async_pipeline(<some params>))
    # Execute all sub pipelines in parallel
    [p.result() for p in submitted_pipelines]

teach()
engine.shutdown()
