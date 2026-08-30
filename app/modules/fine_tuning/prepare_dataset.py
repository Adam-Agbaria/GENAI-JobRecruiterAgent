"""
Turns sms_conversations.json into an OpenAI fine-tuning JSONL for the
Conversation Exit Advisor: for every labeled recruiter turn, reconstruct the
history strictly before it and a binary is_end target (label == "end").

Note: with only 15 positive ("end") examples across 59 labeled turns, this is
a genuinely small fine-tuning set. The pipeline is still run as a real
attempt (see run_finetune.py); the Exit Advisor keeps working via few-shot
prompting regardless of the outcome (app/modules/exit_advisor/m_2.py).
"""
import json

from app import config
from app.conversation import format_history
from app.modules.exit_advisor.m_2 import SYSTEM_PROMPT

SOURCE_PATH = str(config.PROJECT_ROOT / "sms_conversations.json")
OUTPUT_PATH = str(config.PROJECT_ROOT / "data" / "exit_advisor_finetune.jsonl")


def build_examples(source_path: str = SOURCE_PATH) -> list[dict]:
    with open(source_path, encoding="utf-8") as f:
        conversations = json.load(f)

    examples = []
    for conv in conversations:
        turns = conv["turns"]
        for i, turn in enumerate(turns):
            if turn["speaker"] != "recruiter" or turn.get("label") is None:
                continue

            history = turns[:i]
            candidate_message = next(
                (t["text"] for t in reversed(history) if t["speaker"] == "candidate"),
                "(conversation just started)",
            )
            is_end = turn["label"] == "end"

            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Conversation so far:\n{format_history(history)}\n\n"
                                f"Candidate's latest message: {candidate_message}"
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "should_end": is_end,
                                    "confidence": 0.9 if is_end else 0.1,
                                    "reason": "labeled training example",
                                }
                            ),
                        },
                    ]
                }
            )
    return examples


def write_jsonl(examples: list[dict], output_path: str = OUTPUT_PATH) -> int:
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return len(examples)


if __name__ == "__main__":
    examples = build_examples()
    positive = sum(
        1 for ex in examples if json.loads(ex["messages"][-1]["content"])["should_end"]
    )
    count = write_jsonl(examples)
    print(f"Wrote {count} examples ({positive} positive 'end' examples) to {OUTPUT_PATH}")
