"""
Evaluation script — measures pipeline accuracy on known test cases.
Reports precision, recall, F1 score and saves results to eval/results.csv
"""

import os
import sys
import csv
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from app.pipeline import HallucinationDetectionPipeline
from app.schemas import CheckRequest

logging.basicConfig(level=logging.WARNING)  # suppress pipeline logs during eval

EVAL_INPUT  = Path("eval/test_cases.csv")
EVAL_OUTPUT = Path("eval/results.csv")


def run_evaluation():
    print("\n🔍 Hallucination Detector — Evaluation")
    print("=" * 50)

    # Load test cases
    df = pd.read_csv(EVAL_INPUT)
    print(f"Loaded {len(df)} test cases\n")

    pipeline = HallucinationDetectionPipeline()

    results = []
    correct = 0

    for _, row in df.iterrows():
        test_id       = row["id"]
        text          = row["text"]
        expected      = row["expected_verdict"]

        print(f"[{test_id:02d}] Testing: {text[:60]}...")

        start = time.time()
        try:
            req    = CheckRequest(text=text)
            result = pipeline.run(req)

            # Use the first annotation's verdict as the predicted verdict
            predicted = result.annotations[0].verdict if result.annotations else "unverifiable"
            confidence = result.annotations[0].confidence if result.annotations else 0.0
            elapsed = round(time.time() - start, 2)

            # Treat "unverifiable" as neither correct nor incorrect — skip in metrics
            if predicted == "unverifiable":
                status = "SKIP"
            elif predicted == expected:
                status = "CORRECT"
                correct += 1
            else:
                status = "WRONG"

            icon = {"CORRECT": "✅", "WRONG": "❌", "SKIP": "❓"}.get(status, "?")
            print(f"      {icon} Expected: {expected:<15} Got: {predicted:<15} ({elapsed}s)")

        except Exception as e:
            predicted  = "error"
            confidence = 0.0
            elapsed    = round(time.time() - start, 2)
            status     = "ERROR"
            print(f"      ⚠️  Error: {e}")

        results.append({
            "id":         test_id,
            "text":       text,
            "expected":   expected,
            "predicted":  predicted,
            "confidence": round(confidence, 4),
            "status":     status,
            "elapsed_s":  elapsed,
        })

    # ── Compute metrics ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    # Filter out skips and errors for metric calculation
    scored = results_df[results_df["status"].isin(["CORRECT", "WRONG"])]

    # Precision, Recall, F1 for "contradicted" class (hallucination detection)
    tp = len(scored[(scored["predicted"] == "contradicted") & (scored["expected"] == "contradicted")])
    fp = len(scored[(scored["predicted"] == "contradicted") & (scored["expected"] == "supported")])
    fn = len(scored[(scored["predicted"] == "supported")    & (scored["expected"] == "contradicted")])
    tn = len(scored[(scored["predicted"] == "supported")    & (scored["expected"] == "supported")])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(scored) if len(scored) > 0 else 0.0

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total test cases : {len(df)}")
    print(f"Scored           : {len(scored)}")
    print(f"Skipped          : {len(results_df[results_df['status'] == 'SKIP'])}")
    print(f"Errors           : {len(results_df[results_df['status'] == 'ERROR'])}")
    print()
    print(f"True Positives   : {tp}  (correctly caught hallucinations)")
    print(f"False Positives  : {fp}  (true facts flagged as hallucinations)")
    print(f"True Negatives   : {tn}  (correctly verified true facts)")
    print(f"False Negatives  : {fn}  (missed hallucinations)")
    print()
    print(f"Accuracy         : {accuracy:.1%}")
    print(f"Precision        : {precision:.1%}")
    print(f"Recall           : {recall:.1%}")
    print(f"F1 Score         : {f1:.1%}")
    print("=" * 50)

    # ── Save results ─────────────────────────────────────────────────────────
    EVAL_OUTPUT.parent.mkdir(exist_ok=True)
    results_df.to_csv(EVAL_OUTPUT, index=False)
    print(f"\n💾 Detailed results saved to {EVAL_OUTPUT}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    run_evaluation()