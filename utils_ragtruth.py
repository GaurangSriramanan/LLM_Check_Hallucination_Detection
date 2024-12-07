from collections import defaultdict

import jsonlines
import torch
from fastchat.model import get_conversation_template
from tqdm import tqdm


from common_utils import *


def get_ragtruth_data(n_samples=200):
    train_data = []
    test_data = []
    with jsonlines.open("rag_truth/dataset/response_summary.jsonl") as f:
        for row in f:
            if row["model"] == "llama-2-7b-chat" and row["split"] == "train":
                train_data.append(row)
            elif row["model"] == "llama-2-7b-chat" and row["split"] == "test":
                test_data.append(row)
            if len(train_data) >= n_samples:
                break
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
        # define the prompt, response and labels as per the dataset
        prompt = data[i]["prompt"]
        response = data[i]["response"]
        label = 1 if data[i]["halluc_count"] > 0 else 0
        labels.append(label)

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
