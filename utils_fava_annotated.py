import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from fastchat.model import get_conversation_template
from tqdm import tqdm

from common_utils import *

_TAGS = ["entity", "relation", "sentence", "invented", "subjective", "unverifiable"]


def get_modified_data():
    # loading "annotations.json" file
    with open("annotations.json", "r", encoding="utf-8") as f:
        data = json.loads(f.read())

    df = {
        "prompt": [],
        "output": [],
        "annotated": [],
        "modified": [],
        "model": [],
        "entity": [],
        "relation": [],
        "sentence": [],
        "invented": [],
        "subjective": [],
        "unverifiable": [],
        "hallucinated": [],
    }

    def modify(s):
        indicator = [0, 0, 0, 0, 0, 0]
        soup = BeautifulSoup(s, "html.parser")
        s1 = ""
        for t in range(len(_TAGS)):
            indicator[t] = len(soup.find_all(_TAGS[t]))
        for elem in soup.find_all(text=True):
            if elem.parent.name != "delete":
                s1 += elem
        return s1, indicator

    for i in range(len(data)):
        df["prompt"].append(data[i]["prompt"])
        df["output"].append(data[i]["output"])
        df["annotated"].append(data[i]["annotated"])
        df["model"].append(data[i]["model"])
        modified_text, indicator = modify(data[i]["annotated"])
        df["modified"].append(modified_text)
        for t in range(len(_TAGS)):
            df[_TAGS[t]].append(indicator[t])
        df["hallucinated"].append(int(sum(indicator) > 0))

    df = pd.DataFrame(df)
    return df


def get_fava_data(n_samples=200):
    np.random.seed(0)
    data = get_modified_data()
    i1 = data["hallucinated"] == 1
    i2 = data["hallucinated"] == 0
    df = pd.concat(
        [
            data[i1][:n_samples],
            data[i2][:n_samples],
        ],
        ignore_index=True,
        sort=False,
    )
    return df, []


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
        row = data.loc[i]

        # define the prompt, response and labels as per the dataset
        prompt = row["prompt"]
        response = row["output"]
        labels.append(1 if row["hallucinated"] == 1 else 0)

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
