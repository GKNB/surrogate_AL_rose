python bootstrap.py --pipeline_dir /pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/experiment/test_03 --input_data_dir /pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/surrogate_AL/data/

python train.py --model gpr --config /pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/config/gpr.yaml --iteration 1 --pipeline_dir /pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/experiment/test_03

python active.py --model gpr --config /pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/config/gpr.yaml --iteration 1 --pipeline_dir /pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/experiment/test_03 --n_new_samples 100

python train.py --model gpr --config /pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/config/gpr.yaml --iteration 2 --pipeline_dir /pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/experiment/test_03

python active.py --model gpr --config /pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/config/gpr.yaml --iteration 2 --pipeline_dir /pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/experiment/test_03 --n_new_samples 100





python bootstrap.py --pipeline_dir /pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/experiment/test_04 --input_data_dir /pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/surrogate_AL/data/

python active.py --model bnn --config /pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/config/gpr.yaml --iteration 2 --pipeline_dir /pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/experiment/test_04 --n_new_samples 100
