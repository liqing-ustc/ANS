# Learn 1 meaning
# Learn syntax
export OUTPUT=./outputs/exp_perception_semantics_curriculum/ 
mkdir $OUTPUT
python train.py --perception --semantics --curriculum --epochs=50 --output-dir=$OUTPUT >$OUTPUT/main.log

# Learn perception
export OUTPUT=./outputs/exp_syntax_semantics_curriculum/ 
mkdir $OUTPUT
python train.py --syntax --semantics --curriculum --epochs=50 --output-dir=$OUTPUT >$OUTPUT/main.log

# Learn semantics
export OUTPUT=./outputs/exp_perception_syntax_curriculum/ 
mkdir $OUTPUT
python train.py --perception --syntax --curriculum --epochs=10 --output-dir=$OUTPUT >$OUTPUT/main.log


# Learn 2 meanings
# Learn perception and syntax
export OUTPUT=./outputs/exp_semantics_curriculum/ 
mkdir $OUTPUT
python train.py --semantics --curriculum --epochs=100 --output-dir=$OUTPUT >$OUTPUT/main.log

# Learn syntax and semantics
export OUTPUT=./outputs/exp_perception_curriculum/ 
mkdir $OUTPUT
python train.py --perception --curriculum --epochs=100 --output-dir=$OUTPUT >$OUTPUT/main.log

# Learn perception and semantics
export OUTPUT=./outputs/exp_syntax_curriculum/ 
mkdir $OUTPUT
python train.py --syntax --curriculum --epochs=100 --output-dir=$OUTPUT >$OUTPUT/main.log

# Learn 3 meanings
export OUTPUT=./outputs/exp_curriculum/ 
mkdir $OUTPUT
python train.py --curriculum --epochs=500 --output-dir=$OUTPUT >$OUTPUT/main.log