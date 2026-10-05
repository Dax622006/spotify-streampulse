# spotify-streampulse/scripts/run_pipeline.py
"""
CLI entry point to execute the complete StreamPulse ETL and model training pipeline.
"""
import sys
import logging
from streampulse.logging_setup import setup_logging
from streampulse.orchestration import run_all

def main():
    setup_logging()
    print("=" * 60)
    print("Spotify StreamPulse - Pipeline Execution")
    print("=" * 60)
    try:
        summary = run_all()
        print("\n[SUCCESS] Pipeline executed successfully!")
        print(f"Timestamp:       {summary['run_timestamp']}")
        print(f"Duration:        {summary['duration_seconds']}s")
        print(f"Source SHA-256:  {summary['source_sha256'][:16]}...")
        print(f"Warehouse Hash:  {summary['warehouse_content_hash'][:16]}...")
        print(f"Registered:      {', '.join(summary['models_registered'])}")
        return 0
    except Exception as exc:
        print(f"\n[FAIL] Pipeline failed: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
