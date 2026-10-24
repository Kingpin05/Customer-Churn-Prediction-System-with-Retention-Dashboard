"""Churn Watch web app: five screens on top of the trained model's artifacts."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import threading
from pathlib import Path

import pandas as pd
from flask import Flask, Response, abort, redirect, render_template, request, url_for

from app.store import OWNERS, STATUSES, TASK_STATUSES, Store
from churn.config import label, load_config, resolve

PAGE_SIZE = 10
_train_lock = threading.Lock()


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, str):
        return v
    return f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:,.3g}"


def cust_label(cid) -> str:
    return f"C-{int(cid):04d}"


def reason_text(item: dict) -> str:
    if item["kind"] == "numeric":
        typical = f" (typical {fmt(item['typical'])})" if item.get("typical") is not None else ""
        return f"{item['label']} is {fmt(item['value'])}{typical}"
    return f"{item['label']} is {fmt(item['value'])}"


def main_reasons(reasons_json: str, k: int = 2) -> str:
    items = [i for i in json.loads(reasons_json) if i["direction"] == "raises"][:k]
    return ", ".join(reason_text(i) for i in items) or "No single factor stands out"


def create_app(
    config_path: str | Path = "config.yaml",
    artifacts_dir: str | Path | None = None,
    db_path: str | Path | None = None,
) -> Flask:
    cfg = load_config(config_path)
    art = Path(artifacts_dir) if artifacts_dir else resolve(cfg, cfg["output_dir"])
    db = Path(db_path) if db_path else resolve(cfg, "var/app.db")
    store = Store(art, db)
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # ---- helpers ---------------------------------------------------------------------------
    def status_of(cid: int, band: str, overrides: dict) -> str:
        o = overrides.get(int(cid))
        if o and o.get("status"):
            return o["status"]
        return "To call" if band in ("High", "Medium") else "No action"

    def row_view(row, overrides: dict) -> dict:
        pct = float(row["risk_percentile"])
        return {
            "id": int(row["customer_id"]),
            "label": cust_label(row["customer_id"]),
            "band": row["band"],
            "top": f"Top {max(1, math.ceil(pct))}%",
            "reasons": main_reasons(row["reasons_json"]),
            "partner": row.get("telecom_partner", "") if "telecom_partner" in row else "",
            "city": row.get("city", "") if "city" in row else "",
            "status": status_of(row["customer_id"], row["band"], overrides),
        }

    def banner() -> dict | None:
        m = store.metrics
        if not m or m["reliable"]:
            return None
        t = m["test"]
        lo, hi = t["roc_auc_range"]
        return {
            "auc": t["roc_auc"],
            "lo": lo,
            "hi": hi,
        }

    @app.context_processor
    def inject():
        m = store.metrics
        return {
            "m": m,
            "banner": banner(),
            "high_count": (m["bands"]["High"] if m else 0),
            "msg": request.args.get("msg", ""),
            "cust_label": cust_label,
            "fmt": fmt,
        }

    def need_model():
        return None if store.ready else render_template("empty.html", page="none")

    def filtered(args) -> tuple[pd.DataFrame, dict]:
        df = store.customers
        overrides = store.followups()
        risk = args.get("risk", "high")
        partner = args.get("partner", "")
        follow = args.get("follow", "")
        q = args.get("q", "").strip()
        if risk == "high":
            df = df[df["band"] == "High"]
        elif risk == "medium":
            df = df[df["band"] == "Medium"]
        if partner and "telecom_partner" in df.columns:
            df = df[df["telecom_partner"] == partner]
        if follow:
            keep = [status_of(i, b, overrides) == follow for i, b in zip(df["customer_id"], df["band"], strict=True)]
            df = df[keep]
        digits = re.sub(r"\D", "", q)
        if q:
            df = df[df["customer_id"] == int(digits)] if digits else df.iloc[0:0]
        return df, {"risk": risk, "partner": partner, "follow": follow, "q": q}

    # ---- screens ---------------------------------------------------------------------------
    @app.get("/")
    def overview():
        if (r := need_model()) is not None:
            return r
        c, m, overrides = store.customers, store.metrics, store.followups()
        n = len(c)
        top = c[c["band"] == "High"].head(6)
        drivers = []
        for s in store.segments["items"][:5]:
            drivers.append(
                {
                    "name": label(cfg, s["feature"]),
                    "range": f"{s['low']['rate']:.0%} to {s['high']['rate']:.0%}",
                    "reliable": s["reliable"],
                }
            )
        return render_template(
            "overview.html",
            page="overview",
            n=n,
            high=m["bands"]["High"],
            medium=m["bands"]["Medium"],
            high_pct=m["bands"]["High"] / n * 100,
            medium_pct=m["bands"]["Medium"] / n * 100,
            rows=[row_view(r, overrides) for _, r in top.iterrows()],
            drivers=drivers,
            cfg=cfg,
            any_reliable=any(s["reliable"] for s in store.segments["items"]),
        )

    @app.get("/customers")
    def customers():
        if (r := need_model()) is not None:
            return r
        df, f = filtered(request.args)
        total = len(df)
        pages = max(1, math.ceil(total / PAGE_SIZE))
        page = min(max(1, request.args.get("page", 1, type=int)), pages)
        chunk = df.iloc[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
        overrides = store.followups()
        partners = (
            sorted(store.customers["telecom_partner"].dropna().unique()) if "telecom_partner" in store.customers else []
        )
        label_for = {"high": "high-risk", "medium": "medium-risk", "all": ""}[f["risk"]]
        return render_template(
            "customers.html",
            page="customers",
            f=f,
            rows=[row_view(r, overrides) for _, r in chunk.iterrows()],
            total=total,
            page_no=page,
            pages=pages,
            start=(page - 1) * PAGE_SIZE + 1 if total else 0,
            end=min(page * PAGE_SIZE, total),
            partners=partners,
            statuses=STATUSES,
            risk_label=label_for,
            medium=store.metrics["bands"]["Medium"],
            high=store.metrics["bands"]["High"],
            query=lambda **kw: url_for("customers", **{**f, **kw}),
        )

    @app.get("/customers.csv")
    def customers_csv():
        if (r := need_model()) is not None:
            return r
        df, _ = filtered(request.args)
        overrides = store.followups()
        buf = io.StringIO()
        w = csv.writer(buf)
        extra = [
            c
            for c in (
                "telecom_partner",
                "gender",
                "age",
                "city",
                "state",
                "num_dependents",
                "estimated_salary",
                "calls_made",
                "sms_sent",
                "data_used",
            )
            if c in df.columns
        ]
        w.writerow(["customer", "risk_rank", "risk_band", "score", "follow_up", "main_reasons", *extra])
        for _, r in df.iterrows():
            w.writerow(
                [
                    cust_label(r["customer_id"]),
                    r["risk_rank"],
                    r["band"],
                    r["score"],
                    status_of(r["customer_id"], r["band"], overrides),
                    main_reasons(r["reasons_json"], 3),
                    *[r[c] for c in extra],
                ]
            )
        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=at-risk-customers.csv"},
        )

    @app.get("/customers/<int:cid>")
    def customer(cid: int):
        if (r := need_model()) is not None:
            return r
        c = store.customers
        if cid not in c.index:
            abort(404)
        row = c.loc[cid]
        overrides = store.followups()
        fu = store.followup(cid) or {}
        items = json.loads(row["reasons_json"])
        reasons = [
            {
                "title": reason_text(i),
                "direction": i["direction"],
                "text": (
                    "Pushes this customer's score up compared with a typical customer."
                    if i["direction"] == "raises"
                    else "Pulls this customer's score down compared with a typical customer."
                ),
            }
            for i in items
        ]
        facts = [
            (lab, fmt(row[col]))
            for lab, col in [
                ("Telecom partner", "telecom_partner"),
                ("Gender", "gender"),
                ("Age", "age"),
                ("City", "city"),
                ("State", "state"),
                ("Dependents", "num_dependents"),
                ("Estimated salary", "estimated_salary"),
                ("Registered", "date_of_registration"),
            ]
            if col in c.columns
        ]
        usage = []
        for lab, col in [
            ("Calls made", "calls_made"),
            ("SMS sent", "sms_sent"),
            ("Data used", "data_used"),
        ]:
            if col in c.columns:
                v, typ = row[col], float(c[col].median())
                top = max(typ, 0 if pd.isna(v) else float(v)) or 1
                usage.append(
                    {
                        "label": lab,
                        "value": fmt(v),
                        "typical": fmt(typ),
                        "w": 0 if pd.isna(v) else max(2, round(float(v) / top * 100)),
                        "tw": round(typ / top * 100),
                    }
                )
        next_step = {
            "High": "Call this week and note what the customer says.",
            "Medium": "Check in by message this month.",
            "Low": "No action needed.",
        }[row["band"]]
        return render_template(
            "customer.html",
            page="customers",
            c=row_view(row, overrides),
            reasons=reasons,
            facts=facts,
            usage=usage,
            next_step=next_step,
            note=fu.get("note", ""),
            statuses=STATUSES,
            outcome="left" if int(row["actual_churned"]) == 1 else "stayed",
            rank=int(row["risk_rank"]),
            total=len(c),
            split=row["split"],
        )

    @app.post("/customers/<int:cid>/status")
    def set_status(cid: int):
        status = request.form.get("status", "")
        if cid not in store.customers.index or status not in STATUSES:
            abort(400)
        store.set_status(cid, status)
        return redirect(url_for("customer", cid=cid, msg=f"Marked as {status.lower()}."))

    @app.post("/customers/<int:cid>/note")
    def set_note(cid: int):
        if cid not in store.customers.index:
            abort(400)
        store.set_note(cid, request.form.get("note", "").strip())
        return redirect(url_for("customer", cid=cid, msg="Note saved."))

    @app.get("/drivers")
    def drivers():
        if (r := need_model()) is not None:
            return r
        seg, tasks = store.segments, store.tasks()
        rows = []
        for i, s in enumerate(seg["items"][:6], 1):
            lab = label(cfg, s["feature"])
            h, lo = s["high"], s["low"]
            action = (
                (
                    cfg["actions"].get(s["feature"])
                    or "Look into why this group leaves more often, then test a retention offer on a small group."
                )
                if s["reliable"]
                else "No action suggested. "
                "This factor does not reliably separate customers who left from those who stayed."
            )
            t = tasks.get(s["feature"], {"owner": OWNERS[0], "status": TASK_STATUSES[0]})
            p = s["p_value"]
            rows.append(
                {
                    "rank": i,
                    "feature": s["feature"],
                    "name": lab,
                    "reliable": s["reliable"],
                    "high": f"Highest: {h['label']} ({h['rate']:.0%} of {h['n']:,} left)",
                    "low": f"Lowest: {lo['label']} ({lo['rate']:.0%} of {lo['n']:,} left)",
                    "verdict": "Real link to leaving"
                    if s["reliable"]
                    else f"Could easily be chance (p = {p:.2f})"
                    if p >= 0.001
                    else "Real link to leaving",
                    "action": action,
                    "owner": t["owner"],
                    "status": t["status"],
                }
            )
        real = [label(cfg, s["feature"]) for s in seg["items"] if s["reliable"]]
        headline = (
            f"{len(real)} factor{'s' if len(real) != 1 else ''} show a real link to leaving: {', '.join(real)}."
            if real
            else "No factor in this data reliably separates customers who left from those who stayed."
        )
        return render_template(
            "drivers.html",
            page="drivers",
            rows=rows,
            headline=headline,
            any_reliable=bool(real),
            owners=OWNERS,
            task_statuses=TASK_STATUSES,
            base=seg["base_rate"],
            n_tests=seg["n_tests"],
        )

    @app.post("/drivers/<feature>")
    def set_driver(feature: str):
        owner, status = request.form.get("owner", ""), request.form.get("status", "")
        if (
            feature not in {s["feature"] for s in store.segments["items"]}
            or owner not in OWNERS
            or status not in TASK_STATUSES
        ):
            abort(400)
        store.set_task(feature, owner, status)
        return redirect(url_for("drivers", msg="Saved."))

    @app.get("/model")
    def model():
        if (r := need_model()) is not None:
            return r
        m = store.metrics
        return render_template(
            "model.html",
            page="model",
            curve=json.dumps(m["curve"]),
            n=m["data"]["rows"],
            default_pct=min(35, 50),
        )

    @app.post("/retrain")
    def retrain():
        from churn.train import run  # imported here so the web app starts fast

        if not _train_lock.acquire(blocking=False):
            return redirect(url_for("model", msg="Training is already running."))
        try:
            run({**cfg, "output_dir": str(store.art)})
            store.load()
            msg = "Retrained on the current data file."
        except Exception as exc:  # noqa: BLE001 - shown to the user, not swallowed
            msg = f"Training failed: {exc}"
        finally:
            _train_lock.release()
        return redirect(url_for("model", msg=msg))

    @app.errorhandler(404)
    def not_found(_):
        return render_template("empty.html", page="none", missing=True), 404

    return app
