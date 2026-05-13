import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from exact_modules.common import answer_score, extract_json, is_nan_like, parse_obj
from exact_modules.config import EXACT_CONFIG
from exact_modules.organizer import build_final_output
from exact_modules.verifier import explanation_score, format_score, reasoning_score


def build_prompt(row):
    prompt = row["prompt"]

    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()

    if isinstance(prompt, list) and len(prompt) > 0:
        if isinstance(prompt[0], dict):
            return str(prompt[0].get("content", ""))

        return str(prompt[0])

    return str(prompt)


def generate_one(model, tokenizer, prompt, max_new_tokens=512):
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def maybe_revise(model, tokenizer, question, final_obj, task_type, gold_answer, max_new_tokens=512):
    if not EXACT_CONFIG.get("use_self_revision_inference", True):
        return final_obj, ""

    from exact_modules.revision.self_revision import build_revision_prompt, build_verifier_feedback

    feedback = build_verifier_feedback(task_type, final_obj, gold_answer=gold_answer)
    revision_prompt = build_revision_prompt(question, final_obj, feedback)

    raw_revision = generate_one(
        model=model,
        tokenizer=tokenizer,
        prompt=revision_prompt,
        max_new_tokens=max_new_tokens,
    )

    obj_revision = extract_json(raw_revision)

    if obj_revision is None:
        return final_obj, raw_revision

    revised = build_final_output(
        model_obj=obj_revision,
        raw_output=raw_revision,
        task_type=task_type,
        question=question,
        extra_info={},
        json_valid_original=True,
    )

    return revised, raw_revision


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default="data/val/exact_rlvr/exact_val_router.parquet")
    parser.add_argument(
        "--output_file",
        type=str,
        default="/content/drive/MyDrive/Explainable_AI/Results/exact_symbolic_moe_predictions.jsonl",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_csv", action="store_true")

    args = parser.parse_args()

    df = pd.read_parquet(args.eval_file)

    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit).reset_index(drop=True)

    print("Eval samples:", len(df))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.eval()

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        extra_info = parse_obj(row["extra_info"])
        task_type = extra_info.get("task_type", row.get("ability", ""))

        reward_model = parse_obj(row["reward_model"])
        gold = reward_model.get("ground_truth", extra_info.get("gold_answer", ""))

        question = extra_info.get("question", "")
        prompt = build_prompt(row)

        raw = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        )

        obj = extract_json(raw)
        json_valid_original = obj is not None

        final_obj = build_final_output(
            model_obj=obj or {},
            raw_output=raw,
            task_type=task_type,
            question=question,
            extra_info=extra_info,
            json_valid_original=json_valid_original,
        )

        revision_raw = ""
        final_obj, revision_raw = maybe_revise(
            model=model,
            tokenizer=tokenizer,
            question=question,
            final_obj=final_obj,
            task_type=task_type,
            gold_answer=gold,
            max_new_tokens=args.max_new_tokens,
        )

        pred_answer = final_obj.get("answer", "")

        p1 = answer_score(pred_answer, gold, task_type)
        p2 = explanation_score(final_obj, raw)
        p3 = reasoning_score(final_obj)
        fmt = format_score(final_obj, raw)

        final_proxy = 0.50 * p1 + 0.25 * p2 + 0.25 * p3

        rows.append({
            "index": extra_info.get("index", ""),
            "task_type": task_type,
            "question": question,
            "gold_answer": gold,
            "pred_answer": pred_answer,
            "raw_output": raw,
            "revision_raw_output": revision_raw,
            "organizer_output": json.dumps(final_obj, ensure_ascii=False),
            "json_valid_original": json_valid_original,
            "format_score": fmt,
            "P1_correctness": p1,
            "P2_explanation_proxy": p2,
            "P3_reasoning_depth_proxy": p3,
            "final_proxy_score": final_proxy,
            "gold_is_valid": not is_nan_like(gold),
        })

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    res = pd.DataFrame(rows)

    if args.save_csv:
        csv_path = out.with_suffix(".csv")
        res.to_csv(csv_path, index=False)
        print("Saved CSV:", csv_path)

    valid = res[res["gold_is_valid"] == True]

    print("\n===== EXACT Symbolic-MoE Evaluation =====")
    print("N:", len(res))
    print("N valid gold:", len(valid))
    print("Answer Accuracy / P1 all:", res["P1_correctness"].mean())

    if len(valid) > 0:
        print("Answer Accuracy / P1 valid gold:", valid["P1_correctness"].mean())

    print("Explanation Proxy / P2:", res["P2_explanation_proxy"].mean())
    print("Reasoning Depth Proxy / P3:", res["P3_reasoning_depth_proxy"].mean())
    print("JSON Valid Original Rate:", res["json_valid_original"].mean())
    print("Format Score:", res["format_score"].mean())
    print("Final Proxy Score:", res["final_proxy_score"].mean())
    print("Saved predictions:", out)


if __name__ == "__main__":
    main()
    