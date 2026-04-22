#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract RoBERTa text features at utterance or token level."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

CFG = {
    'transcription_csv': 'data/transcription.csv',
    'output_root': 'data/features/roberta_large',
    'model_name': 'roberta-large',
    'feature_level': 'UTTERANCE',
    'max_length': 256,
    'batch_size': 16,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'use_fp16': True,
    'skip_if_exists': True,

    'bad_uids': {
        'train_dia125_utt3',
        'val_dia110_utt7',
    },

    'fail_jsonl': '_failures_roberta.jsonl',
}

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def zero_feature(hidden_dim: int, level: str):
    if level == 'UTTERANCE':
        return np.zeros((hidden_dim,), np.float32)
    return np.zeros((1, hidden_dim), np.float32)

@torch.no_grad()
def main():
    df = pd.read_csv(CFG['transcription_csv'])

    level = CFG['feature_level'].upper()
    out_dir = Path(CFG['output_root']) / ('frame' if level == 'FRAME' else 'utt')
    ensure_dir(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(CFG['model_name'], use_fast=True)
    model = AutoModel.from_pretrained(CFG['model_name'])
    model.to(CFG['device']).eval()

    hidden_dim = int(model.config.hidden_size)
    do_fp16 = CFG['use_fp16'] and ('cuda' in CFG['device']) and torch.cuda.is_available()

    rows = df.to_dict(orient='records')
    fails = []

    for s in range(0, len(rows), CFG['batch_size']):
        batch = rows[s:s+CFG['batch_size']]
        names = [str(x['name']) for x in batch]
        texts = [str(x.get('english', '') or '') for x in batch]

        active_names = []
        active_texts = []
        for i, uid in enumerate(names):
            if uid in CFG['bad_uids']:
                fails.append({'uid': uid, 'error': 'skip_bad_uid'})
                continue
            out_path = out_dir / f'{uid}.npy'
            if CFG['skip_if_exists'] and out_path.exists():
                continue
            active_names.append(uid)
            active_texts.append(texts[i])

        if len(active_names) == 0:
            continue

        try:
            enc = tokenizer(
                active_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=CFG['max_length']
            )
            input_ids = enc['input_ids'].to(CFG['device'])
            attention_mask = enc['attention_mask'].to(CFG['device'])

            special_masks = []
            for ids in input_ids.detach().cpu().tolist():
                special_masks.append(tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True))
            special_mask = torch.tensor(special_masks, dtype=torch.bool, device=CFG['device'])

            if do_fp16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )

            last4 = torch.stack(outputs.hidden_states[-4:], dim=0).sum(dim=0)

            for i, uid in enumerate(active_names):
                out_path = out_dir / f'{uid}.npy'
                try:
                    valid = attention_mask[i].bool() & (~special_mask[i])
                    tok = last4[i][valid]

                    if tok.numel() == 0:
                        tok = last4[i][attention_mask[i].bool()]

                    if tok.numel() == 0:
                        arr = zero_feature(hidden_dim, level)
                    else:
                        arr = tok.float().detach().cpu().numpy()
                        if level == 'UTTERANCE':
                            arr = arr.mean(axis=0).astype(np.float32)
                        else:
                            arr = arr.astype(np.float32)

                    np.save(out_path, arr)
                except Exception as e:
                    fails.append({'uid': uid, 'error': f'save_or_pack_failed:{repr(e)}'})
                    continue

        except Exception as e:
            for uid in active_names:
                try:
                    np.save(out_dir / f'{uid}.npy', zero_feature(hidden_dim, level))
                    fails.append({'uid': uid, 'error': f'batch_failed_zero_fallback:{repr(e)}'})
                except Exception as ee:
                    fails.append({'uid': uid, 'error': f'batch_failed_and_zero_save_failed:{repr(ee)}'})
            continue

    if fails:
        with open(Path(CFG['output_root']) / CFG['fail_jsonl'], 'w', encoding='utf-8') as f:
            for x in fails:
                f.write(json.dumps(x, ensure_ascii=False) + '\n')

    print('Done.')

if __name__ == '__main__':
    main()
