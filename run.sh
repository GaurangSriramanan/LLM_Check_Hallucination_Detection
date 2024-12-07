export KMP_DUPLICATE_LIB_OK=True

export HF_HOME="/path/to/hf_home"

# - model can take a value in llama|llama-3|vicuna7b|vicuna13b|pythia|guanaco|mistral|falcon
# - mt can take a values in logit|hidden|attns, where "logit" computes output token uncertainty metrics which are
# perplexity, window entropy and logit entropy; "hidden" method requires SVD explicitly, and is thus a little slower than
# other methods
# - dataset can take a value in fava|fava_annot|selfcheck|rag_truth

python3 -u run_detection_combined.py --model "llama" --mt 'logit' --mt 'attns' --n_samples 500 --dataset 'fava_annot'  | tee -a logs/llama2_7b_chat_hf_fava_annot.txt