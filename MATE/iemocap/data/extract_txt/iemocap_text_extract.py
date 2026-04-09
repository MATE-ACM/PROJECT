# -*- coding: utf-8 -*-



from __future__ import annotations

import os
import json
import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# =========================

# =========================
CONFIG = {
    # 输入 jsonl（每行一个json）
    "JSONL_PATH": r"manifest_iemocap_6way_STRICT.jsonl",

    # 输出目录（每个utt_id一个 .npy）
    "OUT_DIR": r"text_roberta_merbench_FRAME",

   
    "MODEL_NAME": "roberta-large",

    # "UTTERANCE" 输出 [D]；"FRAME" 输出 [T, D]
    "FEATURE_LEVEL": "FRAME",#"UTTERANCE",  # or "FRAME"

    # 截断长度（FRAME时通常更敏感；UTTERANCE也建议固定，保证训练/测试一致）
    "MAX_LENGTH": 256,

    # 批大小：越大越快，但显存更吃紧。roberta-large 推荐 8~32 看显存。
    "BATCH_SIZE": 16,

    # 设备
    "DEVICE": "cuda:0" if torch.cuda.is_available() else "cpu",

    # 是否使用半精度（仅 cuda 有意义）
    "USE_FP16": True,

    # 已存在就跳过（断点续跑）
    "SKIP_IF_EXISTS": True,

    # 打印频率
    "LOG_EVERY": 200,
}


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"JSON parse error at line {ln}: {e}\nLine: {line[:200]}") from e
            if "utt_id" not in obj or "transcript" not in obj:
                raise ValueError(f"jsonl line {ln} missing keys. Need utt_id & transcript. Got: {list(obj.keys())}")
            items.append(obj)
    return items


def _try_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None


@torch.no_grad()
def extract_text_features(
    items: List[Dict[str, Any]],
    out_dir: str,
    model_name: str,
    feature_level: str = "UTTERANCE",
    max_length: int = 256,
    batch_size: int = 16,
    device: str = "cpu",
    use_fp16: bool = True,
    skip_if_exists: bool = True,
    log_every: int = 200,
) -> None:
    feature_level = feature_level.upper().strip()
    if feature_level not in {"UTTERANCE", "FRAME"}:
        raise ValueError(f"FEATURE_LEVEL must be UTTERANCE or FRAME, got: {feature_level}")

    _safe_mkdir(out_dir)

    print("========== CONFIG ==========")
    print(f"model_name     : {model_name}")
    print(f"feature_level  : {feature_level}")
    print(f"max_length     : {max_length}")
    print(f"batch_size     : {batch_size}")
    print(f"device         : {device}")
    print(f"use_fp16       : {use_fp16}")
    print(f"skip_if_exists : {skip_if_exists}")
    print(f"out_dir        : {out_dir}")
    print("============================")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    # fp16 only on cuda
    do_fp16 = use_fp16 and ("cuda" in device) and torch.cuda.is_available()

    tqdm = _try_tqdm()
    iterator = range(0, len(items), batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, total=(len(items) + batch_size - 1) // batch_size, desc="Extract")

    t0 = time.time()
    saved = 0
    skipped = 0

    for bi, start in enumerate(iterator):
        batch = items[start:start + batch_size]
        utt_ids = [x["utt_id"] for x in batch]
        texts = [x.get("transcript", "") if x.get("transcript", "") is not None else "" for x in batch]

        # skip if all exist (fast path)
        if skip_if_exists:
            all_exist = True
            for uid in utt_ids:
                out_path = os.path.join(out_dir, f"{uid}.npy")
                if not os.path.exists(out_path):
                    all_exist = False
                    break
            if all_exist:
                skipped += len(utt_ids)
                continue

        # tokenize
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # special token mask（每个样本一条序列）
        # True means "special token" like CLS/SEP/PAD/<s></s>
        special_masks = []
        input_ids_cpu = input_ids.detach().cpu().tolist()
        for ids in input_ids_cpu:
            sm = tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
            special_masks.append(sm)
        special_mask = torch.tensor(special_masks, dtype=torch.bool, device=device)  # [B, L]

        # forward (hidden_states)
        if do_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) < 4:
            raise RuntimeError("Model did not return enough hidden_states. Check model/config.")

      
        last4 = torch.stack(hidden_states[-4:], dim=0).sum(dim=0)

        # per sample save
        for i, uid in enumerate(utt_ids):
            out_path = os.path.join(out_dir, f"{uid}.npy")
            if skip_if_exists and os.path.exists(out_path):
                skipped += 1
                continue

            # valid = attention_mask == 1 AND not special_token
            valid = attention_mask[i].bool() & (~special_mask[i])

            token_emb = last4[i][valid]  # [T, H]  (T could be 0 if weird text)

            if token_emb.numel() == 0:
                # fallback: at least use non-pad tokens
                valid2 = attention_mask[i].bool()
                token_emb = last4[i][valid2]
                if token_emb.numel() == 0:
                    # ultimate fallback: zeros
                    hdim = last4.shape[-1]
                    if feature_level == "UTTERANCE":
                        arr = np.zeros((hdim,), dtype=np.float32)
                    else:
                        arr = np.zeros((1, hdim), dtype=np.float32)
                    np.save(out_path, arr)
                    saved += 1
                    continue

            if feature_level == "UTTERANCE":
                # mean pooling -> [H]
                emb = token_emb.mean(dim=0).float().detach().cpu().numpy()
                np.save(out_path, emb)
            else:
                # FRAME -> [T, H]
                emb = token_emb.float().detach().cpu().numpy()
                np.save(out_path, emb)

            saved += 1

        if (log_every > 0) and ((start // batch_size) % max(1, log_every // max(1, batch_size)) == 0):
            elapsed = time.time() - t0
            print(f"[progress] {start + len(batch)}/{len(items)} saved={saved} skipped={skipped} elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDone. total={len(items)} saved={saved} skipped={skipped} elapsed={elapsed:.1f}s")
    print(f"Output dir: {out_dir}")


def main():
    items = _read_jsonl(CONFIG["JSONL_PATH"])
    extract_text_features(
        items=items,
        out_dir=CONFIG["OUT_DIR"],
        model_name=CONFIG["MODEL_NAME"],
        feature_level=CONFIG["FEATURE_LEVEL"],
        max_length=int(CONFIG["MAX_LENGTH"]),
        batch_size=int(CONFIG["BATCH_SIZE"]),
        device=str(CONFIG["DEVICE"]),
        use_fp16=bool(CONFIG["USE_FP16"]),
        skip_if_exists=bool(CONFIG["SKIP_IF_EXISTS"]),
        log_every=int(CONFIG["LOG_EVERY"]),
    )


if __name__ == "__main__":
    main()
