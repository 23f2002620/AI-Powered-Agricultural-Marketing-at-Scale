"""
Script 06: End-to-End Pipeline Runner
Runs the complete pipeline from raw CSVs → trained models → campaign plan.
Executes all scripts in order and validates outputs at each stage.

Usage:
    python scripts/06_run_pipeline.py                   # full pipeline
    python scripts/06_run_pipeline.py --stage eda       # single stage
    python scripts/06_run_pipeline.py --crop wheat      # filter to one crop
    python scripts/06_run_pipeline.py --max-growers 100 # limit campaign size
"""

import argparse
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────
STAGES = [
    {
        "name": "eda",
        "script": "scripts/01_eda.py",
        "description": "Exploratory Data Analysis across all 8 tables",
        "required_inputs": ["data/growers.csv", "data/whatsapp_campaign.csv"],
        "expected_outputs": [],
    },
    {
        "name": "feature_store",
        "script": "scripts/02_feature_store.py",
        "description": "Build Grower-360 feature store",
        "required_inputs": ["data/growers.csv", "data/whatsapp_campaign.csv",
                             "data/retailer_inventory_weekly.csv", "data/retailers.csv",
                             "data/reps_territory.csv", "data/retailer_visit_log.csv"],
        "expected_outputs": ["results/grower_feature_store.parquet"],
    },
    {
        "name": "receptivity",
        "script": "scripts/03_train_receptivity.py",
        "description": "Train XGBoost receptivity model (Engine 3)",
        "required_inputs": ["data/growers.csv", "data/whatsapp_campaign.csv",
                             "data/retailers.csv", "data/retailer_inventory_weekly.csv",
                             "data/retailer_visit_log.csv"],
        "expected_outputs": ["models/receptivity_model.pkl"],
    },
    {
        "name": "micro_segmentation",
        "script": "scripts/04_micro_segmentation.py",
        "description": "HDBSCAN micro-segmentation (Engine 4)",
        "required_inputs": ["data/growers.csv", "data/whatsapp_campaign.csv"],
        "expected_outputs": ["results/grower_segments.csv",
                              "results/segment_profiles.parquet",
                              "results/segment_llm_templates.csv"],
    },
    {
        "name": "attribution",
        "script": "scripts/05_attribution_analysis.py",
        "description": "POS attribution analysis (14-day window)",
        "required_inputs": ["data/retailer_pos.csv", "data/retailers.csv",
                             "data/whatsapp_campaign.csv", "data/growers.csv"],
        "expected_outputs": ["results/tehsil_monthly_sales.csv"],
    },
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"


def log(msg, colour=RESET):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{colour}[{ts}] {msg}{RESET}")


def banner(text):
    bar = "─" * 60
    print(f"\n{CYAN}{BOLD}{bar}")
    print(f"  {text}")
    print(f"{bar}{RESET}\n")


def check_inputs(stage: dict) -> bool:
    missing = [p for p in stage["required_inputs"] if not Path(p).exists()]
    if missing:
        log(f"  ✗ Missing inputs for [{stage['name']}]: {missing}", RED)
        return False
    return True


def check_outputs(stage: dict) -> bool:
    missing = [p for p in stage["expected_outputs"] if not Path(p).exists()]
    if missing:
        log(f"  ✗ Missing outputs after [{stage['name']}]: {missing}", YELLOW)
        return False
    return True


def run_stage(stage: dict, python: str = sys.executable) -> bool:
    banner(f"Stage: {stage['name'].upper()}  —  {stage['description']}")

    if not check_inputs(stage):
        log("Skipping stage due to missing inputs.", YELLOW)
        return False

    t0 = time.time()
    result = subprocess.run(
        [python, stage["script"]],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"Stage [{stage['name']}] FAILED (exit {result.returncode}) in {elapsed:.1f}s", RED)
        return False

    ok = check_outputs(stage)
    colour = GREEN if ok else YELLOW
    status = "✓ OK" if ok else "⚠ Outputs missing"
    log(f"Stage [{stage['name']}] {status} — {elapsed:.1f}s", colour)
    return True


def run_campaign_stage(crop: str, max_growers: int):
    banner("Stage: CAMPAIGN PLAN  —  Full AI orchestration (all 4 engines)")
    t0 = time.time()

    try:
        sys.path.insert(0, str(Path(".").resolve()))
        from engines.campaign_orchestrator import CampaignOrchestrator

        orchestrator = CampaignOrchestrator(api_key="")
        results = orchestrator.batch_plan(
            campaign_id="CMP_RABI25_PIPELINE_RUN",
            target_crop=crop if crop != "all" else None,
            max_growers=max_growers,
        )
        elapsed = time.time() - t0
        log(f"Campaign plan: {len(results)} growers targeted in {elapsed:.1f}s", GREEN)

        if len(results) > 0:
            print(f"\n{CYAN}{'─'*60}")
            print("  SAMPLE GENERATED MESSAGES")
            print(f"{'─'*60}{RESET}")
            for _, row in results.head(3).iterrows():
                print(f"\n  Grower: {row.get('grower_id')} | "
                      f"Channel: {row.get('channel')} | "
                      f"Lang: {row.get('language')}")
                print(f"  Score: {row.get('receptivity_score', 0):.4f} | "
                      f"Segment: {row.get('segment_id', 'n/a')}")
                text = str(row.get("content_text", ""))[:200]
                print(f"  Content: {text}...")
                print()

        return True
    except Exception as e:
        elapsed = time.time() - t0
        log(f"Campaign stage FAILED in {elapsed:.1f}s: {e}", RED)
        import traceback
        traceback.print_exc()
        return False


def bootstrap_bandit():
    banner("Stage: BANDIT BOOTSTRAP  —  Warm-start LinUCB from historical WA data")
    t0 = time.time()
    try:
        sys.path.insert(0, str(Path(".").resolve()))
        from engines.targeting_optimizer import bootstrap_from_history

        bandit = bootstrap_from_history(
            "data/whatsapp_campaign.csv",
            "data/growers.csv",
        )
        elapsed = time.time() - t0
        log(f"Bandit bootstrapped: {bandit.total_rounds:,} rounds in {elapsed:.1f}s", GREEN)

        print(f"\n{CYAN}Top 5 arms by empirical reward:{RESET}")
        for stat in bandit.arm_stats()[:5]:
            print(f"  {stat['arm']}: pulls={stat['pulls']}, reward={stat['empirical_reward']}")
        return True
    except Exception as e:
        log(f"Bandit bootstrap FAILED: {e}", RED)
        return False


def print_summary(results: dict):
    banner("PIPELINE SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    colour = GREEN if passed == total else (YELLOW if passed > 0 else RED)
    print(f"{colour}{BOLD}  {passed}/{total} stages completed successfully{RESET}\n")
    for stage, ok in results.items():
        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {icon}  {stage}")
    print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Syngenta Agri-Marketing AI — End-to-End Pipeline Runner"
    )
    parser.add_argument("--stage", default="all",
                        choices=["all"] + [s["name"] for s in STAGES] + ["bandit", "campaign"],
                        help="Run a single stage or 'all'")
    parser.add_argument("--crop", default="all",
                        help="Filter campaign to specific crop (all | wheat | mustard | chickpea | potato)")
    parser.add_argument("--max-growers", type=int, default=100,
                        help="Max growers for campaign plan stage")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA (saves ~1 min)")
    args = parser.parse_args()

    # Create required directories
    for d in ["models", "results", "data"]:
        Path(d).mkdir(exist_ok=True)

    # Check data directory
    if not (Path("data") / "growers.csv").exists():
        log("ERROR: data/growers.csv not found. Copy the dataset CSVs into data/ first.", RED)
        sys.exit(1)

    banner("Syngenta Agri-Marketing AI — Pipeline Runner")
    log(f"Python: {sys.executable}")
    log(f"Stage: {args.stage} | Crop: {args.crop} | Max growers: {args.max_growers}")

    stage_results = {}

    if args.stage == "all":
        for stage in STAGES:
            if args.skip_eda and stage["name"] == "eda":
                log("Skipping EDA (--skip-eda)", YELLOW)
                stage_results[stage["name"]] = True
                continue
            ok = run_stage(stage)
            stage_results[stage["name"]] = ok
            if not ok and stage["name"] in ("feature_store", "receptivity", "micro_segmentation"):
                log(f"Critical stage [{stage['name']}] failed. Halting pipeline.", RED)
                break

        # Bandit bootstrap (after training data stages)
        stage_results["bandit_bootstrap"] = bootstrap_bandit()

        # Campaign planning (all 4 engines)
        stage_results["campaign_plan"] = run_campaign_stage(args.crop, args.max_growers)

    elif args.stage == "bandit":
        stage_results["bandit_bootstrap"] = bootstrap_bandit()

    elif args.stage == "campaign":
        stage_results["campaign_plan"] = run_campaign_stage(args.crop, args.max_growers)

    else:
        matched = next((s for s in STAGES if s["name"] == args.stage), None)
        if matched:
            stage_results[args.stage] = run_stage(matched)
        else:
            log(f"Unknown stage: {args.stage}", RED)
            sys.exit(1)

    print_summary(stage_results)

    failed = [k for k, v in stage_results.items() if not v]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
