"""
Submits and polls an OpenAI fine-tuning job for the Exit Advisor, using the
JSONL produced by prepare_dataset.py. On completion, prints the resulting
model id — copy it into .env as EXIT_ADVISOR_FINE_TUNED_MODEL_ID to activate
it (app/config.py picks it up automatically; no other code changes needed).

Usage:
    python -m app.modules.fine_tuning.prepare_dataset
    python -m app.modules.fine_tuning.run_finetune
"""
import time

from openai import OpenAI

from app import config
from app.modules.fine_tuning.prepare_dataset import OUTPUT_PATH

BASE_MODEL = "gpt-4o-mini-2024-07-18"
POLL_INTERVAL_SECONDS = 20


def run(jsonl_path: str = OUTPUT_PATH, base_model: str = BASE_MODEL) -> str:
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="fine-tune")

    job = client.fine_tuning.jobs.create(training_file=uploaded.id, model=base_model)
    print(f"Submitted fine-tuning job {job.id} (base model {base_model})")

    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(f"  status: {job.status}")
        if job.status in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    if job.status != "succeeded":
        raise RuntimeError(f"Fine-tuning job did not succeed: status={job.status}, error={job.error}")

    print(f"Fine-tuned model id: {job.fine_tuned_model}")
    print("Add this to .env as EXIT_ADVISOR_FINE_TUNED_MODEL_ID to activate it.")
    return job.fine_tuned_model


if __name__ == "__main__":
    run()
