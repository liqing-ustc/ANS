# GRU with symbol input
# export ENC_LAYERS=3
# export DEC_LAYERS=1
# python train.py --perception --enc_layers=$ENC_LAYERS --dec_layers=$DEC_LAYERS >outputs/symbol_enc_${ENC_LAYERS}_dec_${DEC_LAYERS}.log

# GRU with image input
# export ENC_LAYERS=1
# export DEC_LAYERS=1
# python train.py --curriculum --enc_layers=$ENC_LAYERS --dec_layers=$DEC_LAYERS >outputs/enc_${ENC_LAYERS}_dec_${DEC_LAYERS}.log

export ENC_LAYERS=3
export DEC_LAYERS=1
python train.py --curriculum --enc_layers=$ENC_LAYERS --dec_layers=$DEC_LAYERS --epochs=200 >outputs/enc_${ENC_LAYERS}_dec_${DEC_LAYERS}.log

# export ENC_LAYERS=3
# export DEC_LAYERS=3
# python train.py --curriculum --enc_layers=$ENC_LAYERS --dec_layers=$DEC_LAYERS >outputs/enc_${ENC_LAYERS}_dec_${DEC_LAYERS}.log

# export ENC_LAYERS=6
# export DEC_LAYERS=6
# python train.py --curriculum --enc_layers=$ENC_LAYERS --dec_layers=$DEC_LAYERS >outputs/enc_${ENC_LAYERS}_dec_${DEC_LAYERS}.log

# export ENC_LAYERS=6
# export DEC_LAYERS=1
# python train.py --curriculum --enc_layers=$ENC_LAYERS --dec_layers=$DEC_LAYERS >outputs/enc_${ENC_LAYERS}_dec_${DEC_LAYERS}.log

# export ENC_LAYERS=9
# export DEC_LAYERS=1
# python train.py --curriculum --enc_layers=$ENC_LAYERS --dec_layers=$DEC_LAYERS >outputs/enc_${ENC_LAYERS}_dec_${DEC_LAYERS}.log


# # Tranformer with symbol input
# export MODEL=TRAN ENC_LAYERS=1 DEC_LAYERS=1 HEAD=1
# python train.py --seq2seq=${MODEL} --perception --curriculum --enc_layers=$ENC_LAYERS --dec_layers=$DEC_LAYERS --nhead=${HEAD} --epochs=1000 --epochs_eval=100 --dropout=0.1 >outputs/${MODEL}_symbol_enc_${ENC_LAYERS}_dec_${DEC_LAYERS}_head_${HEAD}.log

# export MODEL=TRAN ENC_LAYERS=3 DEC_LAYERS=3 HEAD=4
# python train.py --seq2seq=${MODEL} --perception --curriculum --enc_layers=$ENC_LAYERS --dec_layers=$DEC_LAYERS --nhead=${HEAD} --epochs=1000 --epochs_eval=100 --dropout=0.1 >outputs/${MODEL}_symbol_enc_${ENC_LAYERS}_dec_${DEC_LAYERS}_head_${HEAD}.log

# export MODEL=TRAN ENC_LAYERS=6 DEC_LAYERS=6 HEAD=8
# python train.py --seq2seq=${MODEL} --perception --curriculum --enc_layers=$ENC_LAYERS --dec_layers=$DEC_LAYERS --nhead=${HEAD} --epochs=1000 --epochs_eval=100 --dropout=0.1 >outputs/${MODEL}_symbol_enc_${ENC_LAYERS}_dec_${DEC_LAYERS}_head_${HEAD}.log

# # Tranformer with image input
# export MODEL=TRAN ENC_LAYERS=3 DEC_LAYERS=3 HEAD=4
# python train.py --seq2seq=${MODEL} --curriculum --enc_layers=$ENC_LAYERS --dec_layers=$DEC_LAYERS --nhead=${HEAD} --epochs=1000 --epochs_eval=100 --dropout=0.1 >outputs/${MODEL}_enc_${ENC_LAYERS}_dec_${DEC_LAYERS}_head_${HEAD}.log