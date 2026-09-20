from pathlib import Path

from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REPO_ID = "RiteshKushwaha01/supportiq-artifacts"

ARTIFACTS = [
    "retrieval.index",
    "retrieval_corpus.db",
]


def ensure_artifacts():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for filename in ARTIFACTS:
        destination = PROCESSED_DIR / filename

        if destination.exists():
            print(f"[OK] {filename} already exists")
            continue

        print(f"[DOWNLOAD] {filename}")

        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="model",
            filename=filename,
            local_dir=str(PROCESSED_DIR),
        )

        print(f"[OK] Downloaded {filename}")


if __name__ == "__main__":
    ensure_artifacts()