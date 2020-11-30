# python train.py --curriculum --semantics --output-dir=outputs/exp0/ >outputs/exp0/main.log
# python train.py --semantics --output-dir=outputs/exp1/ >outputs/exp1/main.log
python train.py --curriculum --epochs=500 >outputs/exp2/main.log 2>outputs/exp2/dreamcoder.log