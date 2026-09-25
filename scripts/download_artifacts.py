import os
from pathlib import Path

from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REPO_ID = os.getenv(
    "HF_ARTIFACTS_REPO",
    "RiteshKushwaha01/supportiq-artifacts",
)
REPO_TYPE = os.getenv("HF_ARTIFACTS_REPO_TYPE", "model")

ARTIFACTS = [
    "retrieval.index",
    "retrieval_corpus.db",
]


def ensure_artifacts():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    for filename in ARTIFACTS:
        destination = PROCESSED_DIR / filename

        if destination.exists():
            print(f"[OK] {filename} already exists")
            continue

        print(f"[DOWNLOAD] {filename} from {REPO_ID}")

        hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            local_dir=str(PROCESSED_DIR),
            token=token,
        )

        print(f"[OK] Downloaded {filename}")


if __name__ == "__main__":
    ensure_artifacts()