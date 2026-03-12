"""
appointments.py — Appointment management for Manny v4
======================================================
Stores appointments in data/appointments.json keyed by date.
Daily limit: 5 active appointments (pending + accepted + started).
Finished and declined appointments free up slots.
"""

import json
import os
import uuid
from datetime import date, datetime

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "appointments.json")
MAX_PER_DAY = 5

# ── Status constants ───────────────────────────────────────────────────────────
PENDING  = "pending"    # Customer booked, waiting for mechanic to accept
ACCEPTED = "accepted"   # Mechanic accepted — shows as "Hasn't arrived yet"
STARTED  = "started"    # Car is being worked on
FINISHED = "finished"   # Job done — slot freed
DECLINED = "declined"   # Mechanic declined — slot freed

# Statuses that count toward the daily limit
ACTIVE_STATUSES = {PENDING, ACCEPTED, STARTED}

# Display labels for the mechanic dropdown (after accepting)
STATUS_LABELS = {
    ACCEPTED: "Hasn't arrived yet",
    STARTED:  "Started",
    FINISHED: "Finished",
}
LABEL_TO_STATUS = {v: k for k, v in STATUS_LABELS.items()}

# ── File helpers ───────────────────────────────────────────────────────────────
def _load() -> dict:
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    if not os.path.exists(DATA_FILE):
        return {}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save(data: dict):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def today_key() -> str:
    return date.today().isoformat()

# ── Queries ────────────────────────────────────────────────────────────────────
def get_today_appointments() -> list:
    return _load().get(today_key(), [])

def active_count_today() -> int:
    """Count slots in use (pending + accepted + started)."""
    return sum(1 for a in get_today_appointments() if a["status"] in ACTIVE_STATUSES)

def slots_available() -> int:
    return max(0, MAX_PER_DAY - active_count_today())

def is_full() -> bool:
    return slots_available() == 0

# ── Mutations ──────────────────────────────────────────────────────────────────
def book_appointment(name: str, phone: str, vehicle: str,
                     concern: str, preferred_time: str) -> dict:
    """Create a new pending appointment. Returns the new appointment dict."""
    data = _load()
    key  = today_key()
    if key not in data:
        data[key] = []

    appt = {
        "id":                str(uuid.uuid4())[:8].upper(),
        "name":              name.strip(),
        "phone":             phone.strip(),
        "vehicle":           vehicle.strip(),
        "concern":           concern.strip(),
        "preferred_time":    preferred_time,
        "booked_at":         datetime.now().strftime("%I:%M %p"),
        "status":            PENDING,
        "customer_call_phone": "",   # filled by mechanic when status → started
        "mechanic_notes":    "",
    }
    data[key].append(appt)
    _save(data)
    return appt

def update_appointment(appt_id: str, **kwargs):
    """Update one or more fields on an appointment for today."""
    data = _load()
    key  = today_key()
    for appt in data.get(key, []):
        if appt["id"] == appt_id:
            appt.update(kwargs)
            break
    _save(data)

def get_appointment_by_id(appt_id: str) -> dict | None:
    for appt in get_today_appointments():
        if appt["id"] == appt_id:
            return appt
    return None


def delete_appointment(appt_id: str):
    """Remove an appointment completely from today's data file.

    Used by the mechanic UI when archiving or deleting a finished/declined
    entry. Slots were already freed when the appointment was marked finished
    or declined, so removing it here only affects the view.
    """
    data = _load()
    key = today_key()
    if key not in data:
        return
    original = data[key]
    # filter out the matching id
    new_list = [a for a in original if a.get("id") != appt_id]
    if new_list:
        data[key] = new_list
    else:
        # no appointments left for today, remove the key entirely
        data.pop(key, None)
    _save(data)
