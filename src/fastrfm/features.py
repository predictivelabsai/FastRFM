"""Leakage-safe aggregates.

Every event feature uses rows whose day is strictly less than the anchor day.
An event at noon on day D is visible to anchors after day D and hidden from
the anchor that starts day D. The label windows open at the anchor, so they
do not overlap the features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .warehouse import ORIGIN, as_day

FLAT_CUSTOMER = ["recency_days", "frequency_365", "monetary_365", "tenure_days"]
RFM_CUSTOMER_EXTRA = ["tickets_21", "severity_21", "refund_rate_60", "days_since_severe", "meal_tickets_120"]
RFM_CUSTOMER = FLAT_CUSTOMER + RFM_CUSTOMER_EXTRA

FLAT_PAYMENT = [
    "amount",
    "customer_recency_days",
    "customer_frequency_365",
    "customer_monetary_365",
    "own_prior_payments",
    "own_prior_fraud_rate",
]
RFM_PAYMENT_EXTRA = ["device_prior_n", "device_peer_fraud_rate", "device_n_customers"]
RFM_PAYMENT = FLAT_PAYMENT + RFM_PAYMENT_EXTRA

SENTINEL_DAYS = 365.0


def feature_names(entity: str, mode: str) -> list[str]:
    if entity == "customer":
        return list(RFM_CUSTOMER if mode == "rfm" else FLAT_CUSTOMER)
    if entity == "payment":
        return list(RFM_PAYMENT if mode == "rfm" else FLAT_PAYMENT)
    raise ValueError(entity)


class _Grouped:
    """Rows sorted by key, then by day, with a ``before(key, day)`` slice."""

    def __init__(self, keys: np.ndarray, days: np.ndarray, columns: dict[str, np.ndarray]):
        order = np.lexsort((days, keys))
        self.days = days[order]
        self.cols = {name: values[order] for name, values in columns.items()}
        uniq, starts, counts = np.unique(keys[order], return_index=True, return_counts=True)
        self.start = {int(k): int(s) for k, s in zip(uniq, starts, strict=True)}
        self.count = {int(k): int(c) for k, c in zip(uniq, counts, strict=True)}

    def before(self, key: int, anchor_day: int) -> slice:
        s = self.start.get(int(key))
        if s is None:
            return slice(0, 0)
        c = self.count[int(key)]
        hi = s + int(np.searchsorted(self.days[s : s + c], anchor_day, side="left"))
        return slice(s, hi)


def _days(ts: pd.Series) -> np.ndarray:
    delta = pd.to_datetime(ts) - ORIGIN
    return np.floor(delta / pd.Timedelta(days=1)).to_numpy(np.int32)


@dataclass
class Indexed:
    orders: _Grouped
    tickets: _Grouped
    payments_by_customer: _Grouped
    payments_by_device: _Grouped
    signup: dict[int, int]
    region: dict[int, int]
    payment_lookup: dict[int, tuple[int, int, int, float]]  # customer, device, day, amount

    @classmethod
    def build(cls, tables: dict[str, pd.DataFrame]) -> Indexed:
        orders = tables["orders"]
        tickets = tables["tickets"]
        payments = tables["payments"]
        customers = tables["customers"]
        returned = (orders["status"].to_numpy() == "returned").astype(np.float64)
        order_group = _Grouped(
            orders["customer_id"].to_numpy(np.int64),
            _days(orders["ts"]),
            {
                "amount": orders["amount"].to_numpy(np.float64),
                "net": orders["net_amount"].to_numpy(np.float64),
                "restaurant": orders["restaurant_id"].to_numpy(np.int64),
                "returned": returned,
            },
        )
        ticket_group = _Grouped(
            tickets["customer_id"].to_numpy(np.int64),
            _days(tickets["ts"]),
            {
                "severity": tickets["severity"].to_numpy(np.float64),
                "restaurant": tickets["restaurant_id"].to_numpy(np.int64),
                "meal": (tickets["topic"].to_numpy() == "meal").astype(np.float64),
                "account": (tickets["topic"].to_numpy() == "account").astype(np.float64),
            },
        )
        pay_days = _days(payments["ts"])
        pay_cols = {
            "fraud": payments["is_fraud"].to_numpy(np.float64),
            "customer": payments["customer_id"].to_numpy(np.int64),
            "device": payments["device_id"].to_numpy(np.int64),
            "amount": payments["amount"].to_numpy(np.float64),
        }
        by_customer = _Grouped(payments["customer_id"].to_numpy(np.int64), pay_days, pay_cols)
        by_device = _Grouped(payments["device_id"].to_numpy(np.int64), pay_days, pay_cols)
        signup = {
            int(i): as_day(t)
            for i, t in zip(customers["customer_id"].tolist(), customers["signup_date"].tolist(), strict=True)
        }
        region = {
            int(i): int(r)
            for i, r in zip(customers["customer_id"].tolist(), customers["region"].tolist(), strict=True)
        }
        lookup = {
            int(pid): (int(cid), int(did), int(day), float(amount))
            for pid, cid, did, day, amount in zip(
                payments["payment_id"].tolist(),
                payments["customer_id"].tolist(),
                payments["device_id"].tolist(),
                pay_days.tolist(),
                payments["amount"].tolist(),
                strict=True,
            )
        }
        return cls(order_group, ticket_group, by_customer, by_device, signup, region, lookup)


def _window(days: np.ndarray, sl: slice, anchor_day: int, horizon: int) -> np.ndarray:
    if sl.stop <= sl.start:
        return np.zeros(0, dtype=bool)
    d = days[sl]
    return (d >= anchor_day - horizon) & (d < anchor_day)


def customer_matrix(
    index: Indexed,
    entity_ids: np.ndarray,
    anchors: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    names = feature_names("customer", mode)
    out = np.zeros((len(entity_ids), len(names)), dtype=np.float64)
    for i, (cid, anchor) in enumerate(zip(entity_ids.tolist(), anchors.tolist(), strict=True)):
        day = as_day(anchor)
        out[i] = _customer_vector(index, int(cid), day, mode)
    return out


def _customer_vector(index: Indexed, cid: int, day: int, mode: str) -> np.ndarray:
    osl = index.orders.before(cid, day)
    odays = index.orders.days
    in_year = _window(odays, osl, day, 365)
    n_year = int(in_year.sum())
    if n_year:
        monetary = float(index.orders.cols["amount"][osl][in_year].sum())
        last = int(odays[osl][in_year].max())
        recency = float(day - last)
    else:
        monetary = 0.0
        recency = SENTINEL_DAYS
    signup = index.signup.get(cid, day)
    tenure = float(max(day - signup, 0))
    flat = [recency, float(n_year), monetary, tenure]
    if mode == "flat":
        return np.array(flat, dtype=np.float64)

    tsl = index.tickets.before(cid, day)
    tdays = index.tickets.days
    recent = _window(tdays, tsl, day, 21)
    sev = index.tickets.cols["severity"][tsl]
    account = index.tickets.cols["account"][tsl]
    meal = index.tickets.cols["meal"][tsl]
    account_recent = recent & (account > 0)
    tickets_21 = float(account_recent.sum())
    severity_21 = float(sev[account_recent].sum()) if tickets_21 else 0.0
    meal_120 = _window(tdays, tsl, day, 120) & (meal > 0)
    meal_tickets = float(meal_120.sum())
    severe_days = tdays[tsl][account > 0]
    if len(severe_days):
        days_since = float(day - int(severe_days.max()))
    else:
        days_since = SENTINEL_DAYS

    in_60 = _window(odays, osl, day, 60)
    n_60 = int(in_60.sum())
    if n_60:
        refund_rate = float(index.orders.cols["returned"][osl][in_60].mean())
    else:
        refund_rate = 0.0
    extra = [tickets_21, severity_21, refund_rate, days_since, meal_tickets]
    return np.array(flat + extra, dtype=np.float64)


def payment_matrix(
    index: Indexed,
    payment_ids: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    names = feature_names("payment", mode)
    out = np.zeros((len(payment_ids), len(names)), dtype=np.float64)
    for i, pid in enumerate(payment_ids.tolist()):
        customer, device, day, amount = index.payment_lookup[int(pid)]
        out[i] = _payment_vector(index, customer, device, day, amount, mode)
    return out


def _payment_vector(
    index: Indexed,
    customer: int,
    device: int,
    day: int,
    amount: float,
    mode: str,
) -> np.ndarray:
    # Customer's own orders and payments strictly before this payment's day.
    osl = index.orders.before(customer, day)
    odays = index.orders.days
    in_year = _window(odays, osl, day, 365)
    n_year = int(in_year.sum())
    if n_year:
        monetary = float(index.orders.cols["amount"][osl][in_year].sum())
        recency = float(day - int(odays[osl][in_year].max()))
    else:
        monetary = 0.0
        recency = SENTINEL_DAYS
    psl = index.payments_by_customer.before(customer, day)
    own_n = int(psl.stop - psl.start)
    own_rate = float(index.payments_by_customer.cols["fraud"][psl].mean()) if own_n else 0.0
    flat = [amount, recency, float(n_year), monetary, float(own_n), own_rate]
    if mode == "flat":
        return np.array(flat, dtype=np.float64)

    dsl = index.payments_by_device.before(device, day)
    d_customers = index.payments_by_device.cols["customer"][dsl]
    d_fraud = index.payments_by_device.cols["fraud"][dsl]
    peer = d_customers != customer
    peer_n = int(peer.sum())
    peer_rate = float(d_fraud[peer].mean()) if peer_n else 0.0
    n_distinct = float(len(set(d_customers.tolist()))) if len(d_customers) else 0.0
    extra = [float(dsl.stop - dsl.start), peer_rate, n_distinct]
    return np.array(flat + extra, dtype=np.float64)


def customer_snapshot(index: Indexed, cid: int, anchor: pd.Timestamp) -> dict[str, float]:
    """Named feature values for one customer, both models, at one anchor."""
    day = as_day(anchor)
    flat = _customer_vector(index, cid, day, "flat")
    rfm = _customer_vector(index, cid, day, "rfm")
    out = {name: float(value) for name, value in zip(FLAT_CUSTOMER, flat, strict=True)}
    for name, value in zip(RFM_CUSTOMER_EXTRA, rfm[len(FLAT_CUSTOMER) :], strict=True):
        out[name] = float(value)
    out["region"] = float(index.region.get(cid, -1))
    return out


def notify_keys(
    index: Indexed,
    entity_ids: np.ndarray,
    anchors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieval key (region, recent meal ticket) and the customer's mode restaurant.

    The mode restaurant is the flattened baseline. The key is what the
    in-context model retrieves on.
    """
    keys = np.zeros(len(entity_ids), dtype=np.int64)
    modes = np.full(len(entity_ids), -1, dtype=np.int64)
    for i, (cid, anchor) in enumerate(zip(entity_ids.tolist(), anchors.tolist(), strict=True)):
        day = as_day(anchor)
        osl = index.orders.before(int(cid), day)
        restaurants = index.orders.cols["restaurant"][osl]
        if len(restaurants):
            vals, counts = np.unique(restaurants, return_counts=True)
            modes[i] = int(vals[np.argmax(counts)])
        tsl = index.tickets.before(int(cid), day)
        meal = index.tickets.cols["meal"][tsl]
        recent = _window(index.tickets.days, tsl, day, 120)
        has_meal = bool((recent & (meal > 0)).any())
        region = int(index.region.get(int(cid), 0))
        # Pack region (0..15) and the meal flag into one integer key.
        keys[i] = region + (16 if has_meal else 0)
    return keys, modes
