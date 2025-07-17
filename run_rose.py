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
def bootstrap(*args, pipeline_dir, input_data_dir):
    return Task(executable=f'{code_path}/bootstrap.py --pipeline_dir {pipeline_dir} --input_data_dir {input_data_dir}')
    
#FIXME!! For this two tasks, need to specify if using GPU, and other variables!!
@learner.training_task
def training(*args, model, config_file, iteration, pipeline_dir):
    return Task(executable=f'{code_path}/train.py --model {model} --config {config_file} --iteration {iteration} --pipeline_dir {pipeline_dir}')

@learner.active_learn_task
def active_learning(*args, model, config_file, iteration, pipeline_dir, num_new_samples):
    return Task(executable=f'{code_path}/active.py --model {model} --config {config_file} --iteration {iteration} --pipeline_dir {pipeline_dir} --n_new_samples {num_new_samples}')

def teach_single_pipeline(model, input_data_dir, config_file, pipeline_dir, num_iter, num_new_samples):
    iter_id = 1
    print("Start doing bootstrap (only once!)")
    bs = bootstrap(pipeline_dir=pipeline_dir, input_data_dir=input_data_dir)
    bs.result()
    while iter_id < num_iter:   # n iter means in total n traning and n-1 al
        print(f"Start doing iteration {iter_id}")
        train = training(model=model, config_file=config_file, iteration=iter_id, pipeline_dir=pipeline_dir)
        if iter_id == num_iter - 1:
            train.result()
        else:
            active_learn = active_learning(train, model=model, config_file=config_file, iteration=iter_id, pipeline_dir=pipeline_dir, num_new_samples=num_new_samples)
            active_learn.result()
        iter_id += 1

def teach():
    submitted_pipelines = []
    async_pipeline = learner.as_async(teach_single_pipeline)
    model = "gpr"
    input_data_dir = "/pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/surrogate_AL/data/"
    conf_list = ["/pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/config/gpr.yaml", "/pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/config/gpr_01.yaml"]
    pipeline_dir_list = ["/pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/experiment/test_04", "/pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/experiment/test_05"]
    for config_file, pipeline_dir in zip(conf_list, pipeline_dir_list):
        submitted_pipelines.append(async_pipeline(model=model, input_data_dir=input_data_dir, config_file=config_file, pipeline_dir=pipeline_dir, num_iter=10, num_new_samples=10))
    [p.result() for p in submitted_pipelines]

teach()
engine.shutdown()
