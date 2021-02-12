export seed=0
export CUDA_VISIBLE_DEVICES=0

export OUTPUT=./outputs/exp_curriculum_no_Y/seed_${seed}/
mkdir -p $OUTPUT
python train.py --no_Y --curriculum --output-dir=$OUTPUT --seed=$seed >$OUTPUT/main.log