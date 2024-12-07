from collections import defaultdict
from pathlib import Path

import datasets
import torch
from fastchat.model import get_conversation_template
from tqdm import tqdm

from common_utils import *

datasets.config.DOWNLOADED_DATASETS_PATH = Path("../llm_store/datasets")


def get_selfcheck_data(n_samples=1000):
    dataset = datasets.load_dataset("potsawee/wiki_bio_gpt3_hallucination")
    return dataset["evaluation"], []


def get_scores_dict(model_name_or_path, data, mt_list, args):
    system_prompt = ""
    generation_config = {}
    generation_config.update({"temperature": 0.6, "top_p": 0.9, "top_k": 50, "do_sample": True})
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, dtype=torch.bfloat16, **generation_config)
    tok_lens, labels, tok_ins = [], [], []

    wikibio = datasets.load_dataset("wiki_bio", split="test")

    scores = []
    indiv_scores = {}
    for mt in mt_list:
        indiv_scores[mt] = defaultdict(def_dict_value)

    for i in tqdm(range(len(data))):
        row = data[i]
        wiki_id = row["wiki_bio_test_idx"]
        concept = wikibio[wiki_id]["input_text"]["context"].strip()

        passage_context = ""

        for j, sent in enumerate(row["gpt3_sentences"]):
            # define the prompt, response and labels as per the dataset
            prompt = f"This is a Wikipedia passage about {concept}:" + passage_context
            response = sent

            if row["annotation"][j] == "major_inaccurate":
                label = 1
            elif row["annotation"][j] == "minor_inaccurate":
                label = 0.5
            elif row["annotation"][j] == "accurate":
                label = 0
            else:
                raise ValueError("Invalid annotation")
            labels.append(label)

            # grow passage context over consec sentences for selfcheck passage data
            passage_context = passage_context + " " + sent

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
