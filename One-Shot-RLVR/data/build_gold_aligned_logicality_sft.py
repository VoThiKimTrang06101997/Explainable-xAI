import argparse
import json
import sys
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from exact_modules.logicality_distill_teacher import (
    safe_parse_obj,
    build_gold_aligned_teacher,
    get_premises_nl,
    get_premises_fol,
    compact_list,
)


# ============================================================
# Logic source JSON utilities
# ============================================================

def normalize_text_for_match(x):
    s = str(x or "")
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.replace("−", "-").replace("×", "x")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def as_list_local(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]
    if isinstance(x, tuple):
        return [str(v) for v in x if str(v).strip()]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(v) for v in obj if str(v).strip()]
        except Exception:
            pass
        return [s]
    return [str(x)]


def load_logic_json(path):
    if not path:
        return []

    p = Path(path)
    if not p.exists():
        print("[WARN] logic_json not found:", p)
        return []

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    assert isinstance(data, list), "logic_json must be list[dict] or dict"

    print("Loaded logic_json:", p)
    print("Logic groups:", len(data))
    return data


def build_logic_question_map(logic_data):
    qmap = {}

    for group_idx, item in enumerate(logic_data):
        if not isinstance(item, dict):
            continue

        premises_fol = as_list_local(
            item.get("premises-FOL")
            or item.get("premises_fol")
            or item.get("premises_FOL")
            or []
        )

        premises_nl = as_list_local(
            item.get("premises-NL")
            or item.get("premises_nl")
            or item.get("premises_NL")
            or []
        )

        questions = as_list_local(item.get("questions", []))
        answers = as_list_local(item.get("answers", []))
        explanations = as_list_local(item.get("explanation", []))

        for q_idx, q in enumerate(questions):
            q_key = normalize_text_for_match(q)

            qmap[q_key] = {
                "premises-NL": premises_nl,
                "premises-FOL": premises_fol,
                "answer": answers[q_idx] if q_idx < len(answers) else "",
                "explanation": explanations[q_idx] if q_idx < len(explanations) else "",
                "group_idx": group_idx,
                "question_idx": q_idx,
            }

    print("Logic question map size:", len(qmap))
    return qmap


def lookup_logic_source(question, qmap):
    if not qmap:
        return None

    key = normalize_text_for_match(question)

    if key in qmap:
        return qmap[key]

    for q_key, val in qmap.items():
        if key and (key in q_key or q_key in key):
            return val

    return None


# ============================================================
# Prompt builder
# ============================================================

def build_prompt(question, task_type, extra):
    task_type = str(task_type).lower()

    if task_type == "physics":
        return f"""You are an explainable physics QA system.

Solve the problem using scientific logicality:
1. Problem formalization
2. Evidence generation
3. Model/formula generation
4. Evidence evaluation and unit checking
5. Calculation
6. Conclusion

Return valid JSON only with exactly these keys:
answer, unit, explanation, fol, cot, premises, premises_nl, premises_fol, premises-NL, premises-FOL, confidence.

Problem:
{question}
"""

    premises_nl = compact_list(get_premises_nl(extra), max_items=80, max_chars_each=1000)
    premises_fol = compact_list(get_premises_fol(extra), max_items=80, max_chars_each=1000)

    nl_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises_nl)])
    fol_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises_fol)])

    return f"""You are an explainable logic QA system.

Use both natural-language premises and FOL/rule premises to answer the question.

Logicality pipeline:
1. Problem formalization
2. Evidence generation
3. Evidence evaluation
4. Inference
5. Conclusion

Return valid JSON only with exactly these keys:
answer, explanation, fol, cot, premises, premises_nl, premises_fol, premises-NL, premises-FOL, confidence.

Premises-NL:
{nl_text}

Premises-FOL:
{fol_text}

Question:
{question}
"""


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--output_parquet", type=str, default=None)
    parser.add_argument("--logic_json", type=str, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    logic_data = load_logic_json(args.logic_json)
    logic_qmap = build_logic_question_map(logic_data)

    df = pd.read_parquet(args.input_parquet)
    if args.max_rows:
        df = df.head(args.max_rows)

    rows = []
    stats = {
        "total": 0,
        "physics": 0,
        "logic": 0,
        "logic_matched_json": 0,
        "logic_empty_premises_nl": 0,
        "logic_empty_premises_fol": 0,
        "solver_used": 0,
        "solver_verified": 0,
        "gold_guided": 0,
    }

    for _, row in df.iterrows():
        extra = safe_parse_obj(row.get("extra_info", {}))
        reward_model = safe_parse_obj(row.get("reward_model", {}))

        if not isinstance(extra, dict):
            extra = {}

        if not isinstance(reward_model, dict):
            reward_model = {}

        task_type = str(extra.get("task_type", row.get("ability", ""))).lower()
        question = str(extra.get("question", "") or "")

        gold = (
            reward_model.get("ground_truth")
            or extra.get("gold_answer")
            or extra.get("answer")
            or ""
        )

        if not question or str(gold).strip() == "":
            continue

        # --------------------------------------------------------
        # Critical fix:
        # Inject exact dataset premises-NL/FOL from original logic JSON.
        # These keys are then used by both prompt and teacher target.
        # --------------------------------------------------------
        if task_type == "logic":
            source = lookup_logic_source(question, logic_qmap)
            if source is not None:
                extra["premises-NL"] = source["premises-NL"]
                extra["premises-FOL"] = source["premises-FOL"]
                extra["premises_nl"] = source["premises-NL"]
                extra["premises_fol"] = source["premises-FOL"]
                extra["_source_premises_nl"] = source["premises-NL"]
                extra["_source_premises_fol"] = source["premises-FOL"]
                extra["_source_logic_answer"] = source.get("answer", "")
                extra["_source_logic_explanation"] = source.get("explanation", "")
                stats["logic_matched_json"] += 1

        target_obj = build_gold_aligned_teacher(
            question=question,
            task_type=task_type,
            gold_answer=str(gold),
            extra=extra,
        )

        prompt = build_prompt(question, task_type, extra)
        target = json.dumps(target_obj, ensure_ascii=False)

        if task_type == "logic":
            if len(target_obj.get("premises_nl", [])) == 0:
                stats["logic_empty_premises_nl"] += 1
            if len(target_obj.get("premises_fol", [])) == 0:
                stats["logic_empty_premises_fol"] += 1

        rows.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ],
            "prompt": prompt,
            "target": target,
            "task_type": task_type,
            "question": question,
            "gold_answer": str(gold),
            "teacher_answer": str(target_obj.get("answer", "")),
            "teacher_source": target_obj.get("source", "unknown"),
            "solver_used": bool(target_obj.get("solver_used", False)),
            "solver_verified": bool(target_obj.get("solver_verified", False)),
        })

        stats["total"] += 1
        if task_type in stats:
            stats[task_type] += 1
        if target_obj.get("solver_used", False):
            stats["solver_used"] += 1
        if target_obj.get("solver_verified", False):
            stats["solver_verified"] += 1
        if "gold_guided" in str(target_obj.get("source", "")):
            stats["gold_guided"] += 1

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Saved jsonl:", out_jsonl)

    if args.output_parquet:
        out_parquet = Path(args.output_parquet)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out_parquet, index=False)
        print("Saved parquet:", out_parquet)

    print("Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
    
