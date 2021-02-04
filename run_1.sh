export seed=1
export CUDA_VISIBLE_DEVICES=1

# # Learn 3 meanings
export OUTPUT=./outputs/exp_curriculum/seed_${seed}/
mkdir -p $OUTPUT
python train.py --curriculum --output-dir=$OUTPUT --seed=$seed >$OUTPUT/main.log

# Learn 1 meaning
# Learn syntax
export OUTPUT=./outputs/exp_perception_semantics_curriculum/seed_${seed}/
mkdir -p $OUTPUT
python train.py --perception --semantics --curriculum --output-dir=$OUTPUT --seed=$seed >$OUTPUT/main.log

# Learn perception
export OUTPUT=./outputs/exp_syntax_semantics_curriculum/seed_${seed}/
mkdir -p $OUTPUT
python train.py --syntax --semantics --curriculum --output-dir=$OUTPUT --seed=$seed >$OUTPUT/main.log

# Learn semantics
export OUTPUT=./outputs/exp_perception_syntax_curriculum/seed_${seed}/
mkdir -p $OUTPUT
python train.py --perception --syntax --curriculum --output-dir=$OUTPUT --seed=$seed >$OUTPUT/main.log


# Learn 2 meanings
Learn perception and syntax
export OUTPUT=./outputs/exp_semantics_curriculum/seed_${seed}/
mkdir -p $OUTPUT
python train.py --semantics --curriculum --output-dir=$OUTPUT --seed=$seed >$OUTPUT/main.log

# Learn syntax and semantics
export OUTPUT=./outputs/exp_perception_curriculum/seed_${seed}/
mkdir -p $OUTPUT
python train.py --perception --curriculum --output-dir=$OUTPUT --seed=$seed >$OUTPUT/main.log

# Learn perception and semantics
export OUTPUT=./outputs/exp_syntax_curriculum/seed_${seed}/
mkdir -p $OUTPUT
python train.py --syntax --curriculum --output-dir=$OUTPUT --seed=$seed >$OUTPUT/main.log