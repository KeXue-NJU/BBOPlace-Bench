problem_formulation=dmp
algo=bo

for benchmark in adaptec1 adaptec2 adaptec3 adaptec4 bigblue1 bigblue3
do
python ../src/main.py \
    --name=ISPD2005_HPO_BO_GP \
    --benchmark=${benchmark} \
    --placer=${problem_formulation} \
    --algorithm=${algo} \
    --run_mode=single \
    --n_cpu_max=1 \
    --eval_gp_hpwl=True \
    --n_init=10 \
    --n_sampling_repeat=2 \
    --max_evals=100 \
    --max_eval_time=72 \
    --n_macro=1000000 
done