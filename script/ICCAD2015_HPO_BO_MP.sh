problem_formulation=dmp
algo=bo
benchmark_prefix=superblue

for i in 1 3 4 5 7 10 16 18
do
python ../src/main.py \
    --name=ICCAD2015_HPO_BO_MP \
    --benchmark=${benchmark} \
    --placer=${problem_formulation} \
    --algorithm=${algo} \
    --run_mode=single \
    --n_cpu_max=1 \
    --eval_gp_hpwl=False \
    --n_init=10 \
    --n_sampling_repeat=2 \
    --max_evals=200 \
    --max_eval_time=72 \
    --n_macro=512 
done