# export ENC_LAYERS=1
# export DEC_LAYERS=1
# python train.py --curriculum --enc_layers=$ENC_LAYERS --dec_layers=$DEC_LAYERS >outputs/enc_${ENC_LAYERS}_dec_${DEC_LAYERS}.log

# export ENC_LAYERS=3
# export DEC_LAYERS=1
# python train.py --curriculum --enc_layers=$ENC_LAYERS --dec_layers=$DEC_LAYERS >outputs/enc_${ENC_LAYERS}_dec_${DEC_LAYERS}.log

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

export ENC_LAYERS=3
export DEC_LAYERS=1
python train.py --perception --enc_layers=$ENC_LAYERS --dec_layers=$DEC_LAYERS >outputs/symbol_enc_${ENC_LAYERS}_dec_${DEC_LAYERS}.log