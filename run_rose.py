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

#FIXME!! I am utility task!!
@learner.simulation_task
def bootstrap(*args, pipeline_dir):
    return Task(executable=f'{code_path}/bootstrap.py --pipeline_dir {pipeline_dir}')
    
#FIXME!! For this two tasks, need to specify if using GPU, and other variables!!
@learner.training_task
def training(*args, model, config_file, iteration, pipeline_dir):
    return Task(executable=f'{code_path}/train.py --model {model} --config {config_file} --iteration {iteration} --pipeline_dir {pipeline_dir}')

@learner.active_learn_task
def active_learning(*args, model, config_file, iteration, pipeline_dir):
    return Task(executable=f'{code_path}/active.py --model {model} --config {config_file} --iteration {iteration} --pipeline_dir {pipeline_dir}')

def teach_single_pipeline(model, config_file, pipeline_dir, num_iter):
    iter_id = 0
    print("Start doing bootstrap (only once!)")
    bs = bootstrap(pipeline_dir=pipeline_dir)
    bs.result()
    while iter_id < num_iter - 1:   # n iter means in total n traning and n-1 al
        print(f"Start doing iteration {iter_id}")
        train = training(model=model, config_file=config_file, iteration=iter_id, pipeline_dir=pipeline_dir)
        active_learn = active_learning(train, model=model, config_file=config_file, iteration=iter_id, pipeline_dir=pipeline_dir)
        active.result()
        iter_id += 1

def teach():
    submitted_pipelines = []
    # Some conf list we set, like seed
    conf_list = []
    async_pipeline = learner.as_async(teach_single_pipeline)
    for model in ["bnn", "gpr"]:
        for config_file in conf_list:
            submitted_pipelines.append(async_pipeline(model=model, config_file=config_file, pipeline_dir=pipeline_dir, num_iter=10))
    [p.result() for p in submitted_pipelines]

teach()
engine.shutdown()
