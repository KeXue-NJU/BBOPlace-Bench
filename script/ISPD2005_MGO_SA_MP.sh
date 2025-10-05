problem_formulation=mgo
algo=sa

for benchmark in adaptec1 adaptec2 adaptec3 adaptec4 bigblue1 bigblue3
do
python ../src/main.py \
    --name=ISPD2005_MGO_SA_MP \
    --benchmark=${benchmark} \
    --placer=${problem_formulation} \
    --algorithm=${algo} \
    --run_mode=single \
    --n_cpu_max=10 \
    --eval_gp_hpwl=False \
    --n_sampling_repeat=250 \
    --max_evals=10000 \
    --max_eval_time=72 \
    --n_macro=1000000 \
    --sampling=random \
    --mutation=shuffle
done