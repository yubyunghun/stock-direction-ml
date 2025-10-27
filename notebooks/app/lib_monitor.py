# app/lib_monitor.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple

def _load_json(p: Path) -> Dict[str, Any]:
    if not p.exists(): return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def load_monitoring(root: Path) -> Tuple[dict, dict, dict]:
    art = root / "artifacts"
    monitor   = _load_json(art / "monitor_snapshot.json")
    paper     = _load_json(art / "paper_trade.json")
    backtest  = _load_json(art / "backtest_summary.json")
    return monitor, paper, backtest

def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur: return default
        cur = cur[k]
    return cur

def summarize_monitor(m: dict) -> dict:
    # Flexible: pull whatever exists
    return {
        "updated_at":        _get(m, "updated_at") or _get(m, "timestamp"),
        "window_days":       _get(m, "window_days") or _get(m, "config", "window_days"),
        "trades_60d":        _get(m, "kpis", "num_trades_60d") or _get(m, "num_trades_60d"),
        "winrate_60d":       _get(m, "kpis", "winrate_60d") or _get(m, "winrate_60d"),
        "avg_prob_60d":      _get(m, "kpis", "avg_prob_60d") or _get(m, "avg_prob_60d"),
        "avg_ret_60d":       _get(m, "kpis", "avg_ret_60d") or _get(m, "avg_ret_60d"),
        "psi_max":           _max_numeric(_get(m, "drift", "psi_by_feature")),
        "ks_max":            _max_numeric(_get(m, "drift", "ks_by_feature")),
    }

def summarize_paper(p: dict) -> dict:
    return {
        "start":      _get(p, "start") or _get(p, "period", "start"),
        "end":        _get(p, "end")   or _get(p, "period", "end"),
        "final_equity": _get(p, "final_equity") or _get(p, "equity", "final"),
        "num_trades": _get(p, "num_trades") or (len(_get(p, "trades", default=[])) if isinstance(_get(p,"trades"), list) else None),
    }

def summarize_backtest(b: dict) -> dict:
    return {
        "fee_bps":        _get(b, "fee_bps"),
        "auc":            _get(b, "metrics", "auc") or _get(b, "auc"),
        "threshold_tau":  _get(b, "threshold", "tau") or _get(b, "tau"),
        "parity_ok":      bool(_get(b, "parity_ok") or _get(b, "parity", "ok") or _get(b, "matched")),
    }

def _max_numeric(d: dict | None):
    if not isinstance(d, dict): return None
    try:
        vals = [float(v) for v in d.values() if v is not None]
        return max(vals) if vals else None
    except Exception:
        return None

def promotion_checks(mon: dict, back: dict) -> dict:
    # Heuristic rules; only enforce if metric is present.
    rules = []
    status = "PASS"
    # R1: winrate >= 0.52 (if we have it)
    wr = mon.get("winrate_60d")
    if wr is not None:
        ok = wr >= 0.52
        rules.append({"rule":"winrate_60d >= 0.52", "value":wr, "pass":ok})
        if not ok: status = "HOLD"
    # R2: psi_max <= 0.2 (drift)
    psi = mon.get("psi_max")
    if psi is not None:
        ok = psi <= 0.2
        rules.append({"rule":"psi_max <= 0.2", "value":psi, "pass":ok})
        if not ok: status = "HOLD"
    # R3: backtest parity ok (if available)
    parity = back.get("parity_ok")
    if parity is not None:
        ok = bool(parity)
        rules.append({"rule":"backtest parity ok", "value":parity, "pass":ok})
        if not ok: status = "HOLD"
    # If no rules were evaluated, mark as UNKNOWN
    if not rules: status = "UNKNOWN"
    return {"status": status, "rules": rules}
