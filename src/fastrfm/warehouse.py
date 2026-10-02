"""Synthetic multi-table warehouse.

The order clock is regular on purpose. A 90-day pause starts on an anchor
day, and the only thing that distinguishes customers who pause from customers
who do not is a burst of support tickets (plus a refund) in the three weeks
before that day. Recency, frequency, and monetary value are statistically
independent of the pause, because earlier pauses are independent of this one.

Fraud lives on a handful of shared devices. A customer's own order history
does not reveal it. Restaurant switches show up as a meal ticket, not as a
change in which restaurant dominates the customer's past orders.

Nothing here is a claim about a real company. It is a warehouse where the
joins are the signal, so a flattened baseline and a relational model can be
compared without an API key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ORIGIN = pd.Timestamp("2023-01-01")
BLOCK = 90
N_DAYS = 720  # eight 90-day blocks, days 0..719
ANCHOR_DAYS = {"train": 360, "val": 450, "test": 540}
N_REGIONS = 6
N_RESTAURANTS = 12
N_RING_DEVICES = 5
N_DEVICES = 40

# Churn: probability a block is preceded by a ticket burst, and the
# probability that burst actually becomes a silent block. A small share of
# blocks go silent with no tickets, so the ticket signal is strong but not
# perfect.
STORM_P = 0.42
STORM_SILENT_P = 0.88
UNEXPLAINED_SILENT_P = 0.08


def day_to_ts(day: int) -> pd.Timestamp:
    """Event time is noon on ``day``, so a midnight anchor excludes that day."""
    return ORIGIN + pd.Timedelta(days=int(day), hours=12)


def anchor_ts(day: int) -> pd.Timestamp:
    return ORIGIN + pd.Timedelta(days=int(day))


def as_day(ts: pd.Timestamp | str) -> int:
    """Floor of the offset from ``ORIGIN``. Noon on day D maps back to D."""
    delta = pd.Timestamp(ts) - ORIGIN
    return int(np.floor(delta / pd.Timedelta(days=1)))


@dataclass
class Warehouse:
    tables: dict[str, pd.DataFrame]
    origin: pd.Timestamp
    anchors: dict[str, pd.Timestamp]
    seed: int
    n_days: int

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name, frame in self.tables.items():
            frame.to_parquet(path / f"{name}.parquet", index=False)
        meta = {
            "source": "fastrfm-synthetic",
            "seed": self.seed,
            "origin": self.origin.isoformat(),
            "n_days": self.n_days,
            "anchors": {k: v.isoformat() for k, v in self.anchors.items()},
            "tables": list(self.tables),
        }
        (path / "source.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> Warehouse:
        path = Path(path)
        meta = json.loads((path / "source.json").read_text())
        tables = {name: pd.read_parquet(path / f"{name}.parquet") for name in meta["tables"]}
        anchors = {k: pd.Timestamp(v) for k, v in meta["anchors"].items()}
        return cls(
            tables=tables,
            origin=pd.Timestamp(meta["origin"]),
            anchors=anchors,
            seed=int(meta["seed"]),
            n_days=int(meta["n_days"]),
        )


def generate(
    n_customers: int = 400,
    *,
    seed: int = 0,
    n_days: int = N_DAYS,
) -> Warehouse:
    """Build customers, restaurants, devices, orders, payments, and tickets."""
    if n_days != N_DAYS:
        raise ValueError(f"n_days is fixed at {N_DAYS} so anchors fall on block boundaries")
    rng = np.random.default_rng(seed)

    restaurants = pd.DataFrame(
        {
            "restaurant_id": np.arange(N_RESTAURANTS, dtype=np.int64),
            "cuisine": [f"cuisine-{i % N_REGIONS}" for i in range(N_RESTAURANTS)],
            "region": np.array([i % N_REGIONS for i in range(N_RESTAURANTS)], dtype=np.int64),
        }
    )
    devices = pd.DataFrame(
        {
            "device_id": np.arange(N_DEVICES, dtype=np.int64),
            "os": ["ios" if i % 2 == 0 else "android" for i in range(N_DEVICES)],
        }
    )

    customer_rows: list[tuple[int, pd.Timestamp, int]] = []
    order_rows: list[tuple] = []
    ticket_rows: list[tuple] = []
    # customer_id -> list of (order_id, day) so refunds can mark a real order
    orders_by_customer: dict[int, list[tuple[int, int]]] = {}
    order_id = 0
    ticket_id = 0
    n_blocks = n_days // BLOCK

    for cid in range(n_customers):
        region = int(rng.integers(0, N_REGIONS))
        signup = int(rng.integers(0, 40))
        customer_rows.append((cid, day_to_ts(signup), region))
        favorite = region
        alt = (region + 1) % N_REGIONS
        avoid_start = int(rng.integers(160, 560)) if rng.random() < 0.80 else None
        avoid_end = None if avoid_start is None else avoid_start + 120
        if avoid_start is not None:
            ticket_rows.append(
                (
                    ticket_id,
                    cid,
                    -1,
                    favorite,
                    day_to_ts(avoid_start),
                    4,
                    "meal",
                )
            )
            ticket_id += 1

        ticket_blocks: set[int] = set()
        silent_blocks: set[int] = set()
        for b in range(1, n_blocks):
            if rng.random() < STORM_P:
                ticket_blocks.add(b)
                if rng.random() < STORM_SILENT_P:
                    silent_blocks.add(b)
            elif rng.random() < UNEXPLAINED_SILENT_P:
                silent_blocks.add(b)
        for b in sorted(ticket_blocks):
            base = b * BLOCK
            for k in range(3):
                ticket_rows.append(
                    (
                        ticket_id,
                        cid,
                        -1,
                        favorite,
                        day_to_ts(base - 10 + k),
                        5,
                        "account",
                    )
                )
                ticket_id += 1

        cust_orders: list[tuple[int, int]] = []
        day = signup + 6
        while day < n_days:
            if (day // BLOCK) not in silent_blocks:
                in_avoid = avoid_start is not None and avoid_start <= day < avoid_end
                if in_avoid:
                    rid = alt
                elif rng.random() < 0.82:
                    rid = favorite
                else:
                    rid = int(rng.integers(0, N_RESTAURANTS))
                amount = float(rng.lognormal(np.log(32.0), 0.15))
                order_rows.append(
                    (order_id, cid, rid, day_to_ts(day), amount, amount, "placed", day)
                )
                cust_orders.append((order_id, day))
                order_id += 1
            day += 12
        orders_by_customer[cid] = cust_orders

    # The order just before a ticket burst is returned. The row stays, so
    # recency and frequency do not move; only status and net_amount do.
    refund_ids: set[int] = set()
    orders_index: dict[int, list[tuple[int, int]]] = orders_by_customer
    account_days: dict[int, list[int]] = {}
    for row in ticket_rows:
        _tid, cid, _oid, _rid, ts, severity, topic = row
        if topic == "account" and severity >= 5:
            account_days.setdefault(cid, []).append(as_day(ts))
    for cid, days in account_days.items():
        bursts: list[int] = []
        for d in sorted(set(days)):
            if not bursts or d - bursts[-1] > 3:
                bursts.append(d)
        for burst in bursts:
            prior = [pair for pair in orders_index.get(cid, []) if pair[1] < burst]
            if prior:
                refund_ids.add(prior[-1][0])

    patched_orders: list[tuple] = []
    for row in order_rows:
        oid, cid, rid, ts, amount, net, status, day = row
        if oid in refund_ids:
            patched_orders.append((oid, cid, rid, ts, amount, 0.0, "returned", day))
        else:
            patched_orders.append(row)

    # Payments, one per order. Device is the customer's phone unless the
    # payment is routed through a shared (ring) device.
    payment_rows: list[tuple] = []
    for oid, cid, _rid, ts, amount, _net, _status, day in patched_orders:
        home = N_RING_DEVICES + (cid % (N_DEVICES - N_RING_DEVICES))
        if rng.random() < 0.15:
            device = int(rng.integers(0, N_RING_DEVICES))
        else:
            device = home
        payment_rows.append((oid, oid, cid, device, ts, amount, day))

    payment_rows.sort(key=lambda r: (r[6], r[0]))
    ring_n = np.zeros(N_RING_DEVICES, dtype=np.int32)
    ring_fraud = np.zeros(N_RING_DEVICES, dtype=np.int32)
    fraud_flag: dict[int, int] = {}
    for pid, _oid, _cid, device, _ts, _amount, _day in payment_rows:
        if device < N_RING_DEVICES:
            seen = int(ring_n[device])
            prior_rate = float(ring_fraud[device] / seen) if seen else 0.0
            if seen >= 3 and prior_rate >= 0.4:
                is_fraud = int(rng.random() < 0.92)
            elif seen < 2:
                is_fraud = int(rng.random() < 0.55)
            else:
                is_fraud = int(rng.random() < 0.15)
            ring_n[device] = seen + 1
            ring_fraud[device] += is_fraud
        else:
            is_fraud = int(rng.random() < 0.01)
        fraud_flag[pid] = is_fraud

    customers = pd.DataFrame(customer_rows, columns=["customer_id", "signup_date", "region"])
    orders = pd.DataFrame(
        [row[:-1] for row in patched_orders],
        columns=["order_id", "customer_id", "restaurant_id", "ts", "amount", "net_amount", "status"],
    )
    payments = pd.DataFrame(
        [
            (pid, oid, cid, device, ts, amount, fraud_flag[pid])
            for pid, oid, cid, device, ts, amount, _day in payment_rows
        ],
        columns=["payment_id", "order_id", "customer_id", "device_id", "ts", "amount", "is_fraud"],
    )
    tickets = pd.DataFrame(
        ticket_rows,
        columns=["ticket_id", "customer_id", "order_id", "restaurant_id", "ts", "severity", "topic"],
    )
    for frame, col in (
        (customers, "signup_date"),
        (orders, "ts"),
        (payments, "ts"),
        (tickets, "ts"),
    ):
        frame[col] = pd.to_datetime(frame[col])

    tables = {
        "customers": customers,
        "restaurants": restaurants,
        "devices": devices,
        "orders": orders,
        "payments": payments,
        "tickets": tickets,
    }
    anchors = {name: anchor_ts(day) for name, day in ANCHOR_DAYS.items()}
    return Warehouse(tables=tables, origin=ORIGIN, anchors=anchors, seed=seed, n_days=n_days)
