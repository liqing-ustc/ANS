# export FS=0
# export OUTPUT=outputs/exp_fewshot_${FS}/
# mkdir -p $OUTPUT
# python train.py --fewshot=$FS --epochs=10 --output-dir=$OUTPUT >$OUTPUT/main.log

# export FS=1
# export OUTPUT=outputs/exp_fewshot_${FS}/
# mkdir -p $OUTPUT
# python train.py --fewshot=$FS --epochs=10 --output-dir=$OUTPUT >$OUTPUT/main.log

# export FS=2
# export OUTPUT=outputs/exp_fewshot_${FS}/
# mkdir -p $OUTPUT
# python train.py --fewshot=$FS --epochs=10 --output-dir=$OUTPUT >$OUTPUT/main.log

export FS=3
export OUTPUT=outputs/exp_fewshot_${FS}/
mkdir -p $OUTPUT
python train.py --fewshot=$FS --epochs=10 --output-dir=$OUTPUT >$OUTPUT/main.log

# export FS=3
# export OUTPUT=outputs/exp_fewshot_${FS}/
# mkdir -p $OUTPUT
# python train.py --fewshot=$FS --epochs=20 --output-dir=$OUTPUT --resume=$OUTPUT/model_010.p >>$OUTPUT/main.log

# export FS=4
# export OUTPUT=outputs/exp_fewshot_${FS}/
# mkdir -p $OUTPUT
# python train.py --fewshot=$FS --epochs=10 --output-dir=$OUTPUT >$OUTPUT/main.log