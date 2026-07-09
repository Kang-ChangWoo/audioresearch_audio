#!/usr/bin/env python3
"""
utils/research.py - lightweight research-management helper for Auto Audio Depth Estimation.

NOT a framework. A thin, readable layer over the human-editable state files:

  out/results.tsv            - authoritative per-run log (one row per training run; SHORT descriptions)
  out/hypothesis.tsv         - study-level PASS/FAIL conclusions (one line each)
  out/hypothesis_details.tsv - the full general/detailed hypothesis + scientific conclusion per study
  studies.json               - active study state, champion, adaptive-HPO progression, mode, next exp id
  out/ideas.json             - research portfolio (live ideas) + open discrepancies
  out/decision_log.jsonl     - append-only log of meaningful research transitions

Usage (run from repo root):
  python utils/research.py status                                  # print current research state
  python utils/research.py composite --abs_rel A --rmse R --d1 D   # honest composite for a run
  python utils/research.py next-id                                 # print the next experiment id
  python utils/research.py mode <explore|verify|exploit|synthesize> --reason "..."
  python utils/research.py log --event experiment_completed --note "..." [--exp-id E4]

Edit the .tsv/.json files directly for anything this CLI does not cover.
"""
import argparse
import datetime
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
STUDIES = os.path.join(ROOT, "studies.json")
RESULTS = os.path.join(OUT, "results.tsv")
HYP = os.path.join(OUT, "hypothesis.tsv")
HYP_DETAILS = os.path.join(OUT, "hypothesis_details.tsv")
IDEAS = os.path.join(OUT, "ideas.json")
DECISIONS = os.path.join(OUT, "decision_log.jsonl")

MODES = ("explore", "verify", "exploit", "synthesize")

# Mode-aware experiment budget. EXPLORE buys breadth cheaply and must not be allowed to
# turn into HPO; VERIFY/EXPLOIT earn depth by evidence. See program.md for the policy.
BUDGET = {
    "explore":    "1 structural run + 0-2 focused probes -> CANDIDATE / DROP / INCONCLUSIVE",
    "verify":     "clean reimplementation on the correct parent -> standalone run -> bounded HPO -> PASS / FAIL",
    "exploit":    "adaptive HPO ladder 3 -> 5 -> 7 -> 10, each step justified by evidence -> PASS / FAIL",
    "synthesize": "no runs; review evidence, find contradictions, pick the highest-information next question",
}

# Honest-weighted composite (MUST match train.py model selection). Lower is better. RMSE + d1 dominate
# (not directly optimised -> trustworthy); ABS_REL is directly optimisable (gameable) and varies most,
# so it is de-weighted to an effective per-unit coefficient of 0.35 (2026-July).
COMPOSITE_STR = "rmse/1.6 + (1-d1)/0.46 + 0.35*abs_rel"


def composite(abs_rel, rmse, d1):
    return rmse / 1.6 + (1.0 - d1) / 0.46 + 0.35 * abs_rel


def load_studies():
    with open(STUDIES) as f:
        return json.load(f)


def load_ideas():
    if not os.path.exists(IDEAS):
        return {"ideas": [], "discrepancies": []}
    with open(IDEAS) as f:
        return json.load(f)


def _now():
    return datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def log_decision(event, note, mode=None, **extra):
    """Append one meaningful transition to the decision log. Append-only; never rewrite."""
    if mode is None:
        mode = load_studies().get("mode", "exploit")
    rec = {"ts": _now(), "mode": mode, "event": event, **extra, "note": note}
    with open(DECISIONS, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[decision] {event}: {note[:80]}")


def set_mode(mode, reason):
    """Switch research mode. A mode change is only legitimate WITH a reason -- record it."""
    if mode not in MODES:
        raise SystemExit(f"mode must be one of {MODES}")
    st = load_studies()
    old = st.get("mode")
    if old == mode:
        print(f"already in mode '{mode}'")
        return
    st["mode"] = mode
    st["mode_reason"] = reason
    with open(STUDIES, "w") as f:
        json.dump(st, f, indent=2, ensure_ascii=False)
        f.write("\n")
    log_decision("mode_changed", reason, mode=mode, **{"from": old, "to": mode})


def recent_decisions(n=8):
    if not os.path.exists(DECISIONS):
        return []
    with open(DECISIONS) as f:
        lines = [ln for ln in f if ln.strip()]
    out = []
    for ln in lines[-n:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def status():
    st = load_studies()
    ideas = load_ideas()
    gc = st.get("global_champion")
    print("=" * 72)
    print(f"Auto Audio Depth Estimation - research state  [{st.get('phase', '')}]")
    print("=" * 72)
    print(f"metric (lower=better): composite = {COMPOSITE_STR}")
    mode = st.get("mode", "(unset)")
    print(f"MODE            : {mode.upper()}")
    print(f"  budget        : {BUDGET.get(mode, '(legacy study: default to exploit semantics)')}")
    if st.get("mode_reason"):
        print(f"  why           : {st['mode_reason'][:88]}...")
    print("-" * 72)
    if gc:
        print(f"GLOBAL CHAMPION : {gc.get('exp_id')} ({gc.get('lineage')}) commit {gc.get('commit')}")
        print(f"  abs_rel {gc.get('abs_rel')}  rmse {gc.get('rmse')}  d1 {gc.get('d1')}  "
              f"comp {gc.get('composite_mean', gc.get('composite'))}")
    else:
        print("GLOBAL CHAMPION : (none yet)")
        if st.get("reset_note"):
            print(f"  {st['reset_note'][:96]}")
    print("-" * 72)
    a = st["active_study"]
    print(f"ACTIVE STUDY    : {a['study_id']} [{a['type']}] {a['lineage']} (status={a['status']})")
    print(f"  hpo stage     : {a.get('hpo_stage')} (runs used: {a.get('hpo_runs_used', 0)})")
    print(f"  general       : {a['general_hypothesis'][:88]}...")
    print(f"  next exp id   : {st['next_exp_id']}")
    print("-" * 72)
    live = [i for i in ideas.get("ideas", [])
            if i.get("status") not in ("dropped", "validated")]
    print(f"PORTFOLIO ({len(live)} live):")
    for i in live:
        print(f"  {i['id']} [{i['status']}/{i.get('causal_distance','?')}] "
              f"{i.get('mechanism_family','?')} -> {i.get('target_bottleneck','?')[:44]}")
        print(f"      next: {i.get('next_action','?')[:74]}")
    op = [d for d in ideas.get("discrepancies", []) if d.get("status") == "open"]
    if op:
        print("-" * 72)
        print(f"OPEN DISCREPANCIES ({len(op)}):")
        for d in op:
            print(f"  {d['id']}: {d['observation'][:82]}")
    print("-" * 72)
    dec = recent_decisions(5)
    if dec:
        print("RECENT DECISIONS:")
        for d in dec:
            note = d.get("note") or d.get("reason") or ""
            print(f"  {d.get('ts','')[:16]} [{d.get('mode','?')[:4]}] {d.get('event','?')}: {note[:56]}")
    print("-" * 72)
    bl = st.get("backlog", [])
    print(f"BACKLOG ({len(bl)}): " + "; ".join(f"[{b['type']}] {b['lineage']}" for b in bl))
    print("=" * 72)


def main():
    p = argparse.ArgumentParser(description="Auto Audio Depth Estimation research helper")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("status", help="print current research state")
    sub.add_parser("next-id", help="print the next experiment id")
    c = sub.add_parser("composite", help="honest composite for a run")
    c.add_argument("--abs_rel", type=float, required=True)
    c.add_argument("--rmse", type=float, required=True)
    c.add_argument("--d1", type=float, required=True)

    m = sub.add_parser("mode", help="switch research mode (records the reason)")
    m.add_argument("mode", choices=MODES)
    m.add_argument("--reason", required=True)

    lg = sub.add_parser("log", help="append a decision to out/decision_log.jsonl")
    lg.add_argument("--event", required=True,
                    help="study_opened | experiment_completed | candidate_promoted | "
                         "candidate_dropped | verification_started | hypothesis_concluded | "
                         "mode_changed | divergence_checkpoint | discrepancy_recorded | direction_changed")
    lg.add_argument("--note", required=True)
    lg.add_argument("--exp-id", dest="exp_id", default=None)
    lg.add_argument("--idea-id", dest="idea_id", default=None)

    args = p.parse_args()

    if args.cmd == "composite":
        print(f"{composite(args.abs_rel, args.rmse, args.d1):.4f}")
    elif args.cmd == "next-id":
        print(load_studies()["next_exp_id"])
    elif args.cmd == "mode":
        set_mode(args.mode, args.reason)
    elif args.cmd == "log":
        extra = {k: v for k, v in (("exp_id", args.exp_id), ("idea_id", args.idea_id)) if v}
        log_decision(args.event, args.note, **extra)
    else:
        status()


if __name__ == "__main__":
    main()
