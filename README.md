# Hallucination Detection

The run_detection_combined.py script is used to detect hallucinations using the types of scores specified.

For instance, to run on the FAVA Annotations dataset, please download the json file hosted at https://huggingface.co/datasets/fava-uw/fava-data/blob/main/annotations.json, and save it at the top-level directory. 

The 'logit' and 'attns' score methods run very quickly, while the 'hidden' score is slower and waits for SVD to finish explicitly.

The dataset specific utils such as utils_selfcheck.py are used to load the dataset loading function and the dataset-specific iterators which will internally call the score computations. 

The final scores are saved in the /data folder, and can be analyzed using the check_scores.ipynb file to analyze the scores, plot etc. This can be done without a GPU as the scores are already computed and saved. 