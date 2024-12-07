import json

import torch
from collections import defaultdict
from fastchat.model import get_conversation_template
from tqdm import tqdm

from common_utils import *


def get_fava_data(n_samples=1000):
    # need to process the FAVA Train data at https://huggingface.co/datasets/fava-uw/fava-data
    # need to remove tags and annotations to create pairs of samples with and without hallucinations
    with open("processed_fava_text.json") as f:
        data = json.load(f)
        train_data = data[: n_samples // 2]  # each sample has 2 responses, so we only need half of the samples
        test_data = data[n_samples // 2 : n_samples]
    return train_data, test_data


def get_scores_dict(model_name_or_path, data, mt_list, args):
    system_prompt = ""
    generation_config = {}
    generation_config.update({"temperature": 0.6, "top_p": 0.9, "top_k": 50, "do_sample": True})
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, dtype=torch.bfloat16, **generation_config)
    tok_lens, labels, tok_ins = [], [], []

    scores = []
    indiv_scores = {}
    for mt in mt_list:
        indiv_scores[mt] = defaultdict(def_dict_value)

    for i in tqdm(range(len(data))):
        for j in ["0", "1"]:  # labels for the absence/presence of hallucinations
            # define the prompt, response and labels as per the dataset
            prompt = data[i]["prompt"]
            rind = prompt.rfind(
                "Please identify all the errors in the following passage using the references provided and suggest edits"
            )
            prompt = prompt[:rind] + "Based only on the references provided, write a blog article."
            # remove the FAVA prompt to edit, and insert back the original prompt to write an article

            response = data[i][j]
            labels.append(1 if j == "1" else 0)

            chat_template = get_conversation_template(model_name_or_path)
            chat_template.set_system_message(system_prompt.strip())
            chat_template.append_message(chat_template.roles[0], prompt.strip())
            chat_template.append_message(chat_template.roles[1], response.strip())

            full_prompt = chat_template.get_prompt()
            user_prompt = full_prompt.split(response.strip())[0].strip()

            tok_in_u = tokenizer(user_prompt, return_tensors="pt", add_special_tokens=True).input_ids
            tok_in = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=True).input_ids
            tok_lens.append([tok_in_u.shape[1], tok_in.shape[1]])

            logit, hidden_act, attn = get_model_vals(model, tok_in.to(0))
            # Unpacking the values into lists on CPU
            logit = logit[0].cpu()
            hidden_act = [x[0].to(torch.float32).detach().cpu() for x in hidden_act]
            attn = [x[0].to(torch.float32).detach().cpu() for x in attn]
            tok_in = tok_in.cpu()

            tok_len = [tok_in_u.shape[1], tok_in.shape[1]]
            compute_scores(
                [logit],
                [hidden_act],
                [attn],
                scores,
                indiv_scores,
                mt_list,
                [tok_in],
                [tok_len],
                use_toklens=args.use_toklens,
            )

    return scores, indiv_scores, labels
