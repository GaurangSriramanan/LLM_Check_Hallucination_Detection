export KMP_DUPLICATE_LIB_OK=True

export TRANSFORMERS_CACHE="path/to/transformers/cache"


python3 -u run_detection_combined.py --model "llama" --mt 'logit' --mt 'attns' --n_samples 500 --dataset 'fava_annot'  | tee -a logs/llama_fava_annot.txt