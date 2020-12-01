# python train.py --curriculum --semantics --output-dir=outputs/exp0/ >outputs/exp0/main.log
# python train.py --semantics --output-dir=outputs/exp1/ >outputs/exp1/main.log

export OUTPUT=./outputs/exp_syntax_semantics_curriculum/ 
mkdir $OUTPUT
python train.py --syntax --semantics --curriculum --epochs=100 --output-dir=$OUTPUT >$OUTPUT/main.log

export OUTPUT=./outputs/exp_perception_semantics_curriculum/ 
mkdir $OUTPUT
python train.py --perception --semantics --curriculum --epochs=100 --output-dir=$OUTPUT >$OUTPUT/main.log

# export OUTPUT=./outputs/exp2/ 
# python train.py --curriculum --epochs=500 --output-dir=$OUTPUT >$OUTPUT/main.log 2>$OUTPUT/dreamcoder.log