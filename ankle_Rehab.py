"""
============================================================
ANKLE REHABILITATION GUIDANCE & MONITORING SYSTEM
Version 3.1 - Hardware-first patient mode, per-repetition storage

Matches the EXISTING ESP32 firmware output. Firmware unchanged.

Firmware line format (100 Hz, 115200 baud):

    Foot R/P/Y: -1.2 / 3.4 / 5.6   |   Shank R/P/Y: 1.1 / 2.2 / 3.3   |   Ankle: 20.15 deg

Firmware commands:

    'N'  ->  capture neutral position (blocks ~2 s, prints a banner)

Data flow (UNCHANGED from v2.0):

    Two MPU9250 (foot + shank)
        v
    ESP32 : Madgwick fusion -> qFoot, qShank
        v
    ESP32 : qRelative = conj(qShank) * qFoot
        v
    ESP32 : qMotion   = conj(qNeutral) * qRelative
        v
    Serial text line (Foot RPY, Shank RPY, Ankle roll angle)
        v
    Python : parse -> rebuild quaternions from Euler
        v
    Python : select movement axis per exercise
        v
    Python : ROM / repetition / correctness
        v
    Streamlit GUI -> SQLite + CSV

WHAT CHANGED IN v3.0 (everything else is identical to v2.0):

    1. Patient / Developer mode toggle (technical panels hidden from patients)
    2. Every attempt above MIN_PEAK_ATTEMPT is recorded. Under-range attempts
       are stored as INCORRECT instead of being invisible.
    3. New `repetitions` table: one row per attempt (additive, `sessions`
       table and CSV format are untouched).
    4. Reaching the target count of correct reps = "Completed" always.
       Accuracy (correct / attempted) is shown separately.
    5. Completion panel with Start Next Exercise / Finish Session.
    6. Balloons fire once (flag), not on every rerun.
    7. Dashboard reads totals from SQLite instead of session state.
    8. Exercise page: tutorial video slot, rotating motivational quote.
    9. Small CSS block for colour and a single completion animation.
============================================================
"""

import io
import math
import random
import re
import sqlite3
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

# pyserial is optional so that simulation mode still runs on a machine
# without it installed.
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except Exception:
    SERIAL_AVAILABLE = False


# ============================================================
# CONSTANTS
# ============================================================

DB_NAME = "ankle_rehabilitation.db"

# The firmware runs its loop at a hard 100 Hz. Timestamps are derived
# from the sample index rather than host clock, because a single
# Streamlit rerun drains many samples at once and they would otherwise
# all share the same host timestamp.
FIRMWARE_RATE_HZ = 100.0
SAMPLE_PERIOD = 1.0 / FIRMWARE_RATE_HZ

DEFAULT_BAUDRATE = 115200

# Plotting window
PLOT_WINDOW_SECONDS = 20
PLOT_MAX_POINTS = 600

# Serial read cadence inside the Streamlit rerun loop
LOOP_SLEEP = 0.05

# Slower poll used when connected but not recording, so that boot
# banners, device-ready detection and the polarity check still update.
IDLE_SLEEP = 0.15

# Firmware startup: 3 s countdown + 500 gyro samples x 10 ms + banners
DEVICE_BOOT_SECONDS = 11

# NEW in v3.0
# Smallest peak (degrees) that counts as a deliberate attempt. Movements
# below this are treated as sensor noise / drift and are NOT recorded.
# Anything at or above it is recorded, and graded correct or incorrect.
DEFAULT_MIN_PEAK_ATTEMPT = 3.0

# Tutorial videos. Replace any value with your own YouTube link or a local
# path such as "videos/dorsiflexion.mp4" whenever your videos are ready.
# An empty string shows a placeholder instead of a player.
#
# NOTE: These default links come from a web search. They could not be
# opened and verified. Open each one and confirm it plays and shows the
# right movement before demonstrating.
VIDEO_URLS = {
    "Dorsiflexion": "https://www.youtube.com/watch?v=rpTfolKE45o",
    "Plantarflexion": "https://www.youtube.com/watch?v=rpTfolKE45o",
    "Inversion": "https://www.youtube.com/watch?v=n_j4rPMbE7k",
    "Eversion": "https://www.youtube.com/watch?v=n_j4rPMbE7k",
}

MOTIVATION_QUOTES = [
    "Small steps every day lead to big recovery. 💪",
    "Every repetition brings you closer to full strength.",
    "Slow and steady - your ankle is getting stronger.",
    "Consistency beats intensity. Keep showing up!",
    "You're doing something great for your body today.",
    "Progress, not perfection. Well done for starting!",
    "Healing is a journey, and you're already on the road.",
]

# NEW: five different completion effects. One is picked at random each time
# a session is completed, instead of always the same balloon animation.
CELEBRATION_EFFECTS = ["balloons", "snow", "confetti", "fireworks", "sparkles"]


# ============================================================
# SERIAL LINE PARSING  (unchanged)
# ============================================================

NUM = r"([-+]?(?:\d+\.?\d*|\.\d+|nan|inf|-inf|ovf))"

DATA_PATTERN = re.compile(
    r"Foot\s+R/P/Y:\s*" + NUM + r"\s*/\s*" + NUM + r"\s*/\s*" + NUM +
    r"\s*\|\s*Shank\s+R/P/Y:\s*" + NUM + r"\s*/\s*" + NUM + r"\s*/\s*" + NUM +
    r"\s*\|\s*Ankle:\s*" + NUM + r"\s*deg",
    re.IGNORECASE,
)

# Status strings printed by the firmware
MARKER_NEUTRAL_SAVED = "NEUTRAL POSITION SAVED"
MARKER_GYRO_DONE = "GYRO CALIBRATION COMPLETE"
MARKER_READY = "System ready"


def parse_data_line(line):
    """
    Returns dict of the seven floats, or None if the line is not a
    data line (banners, calibration output, garbage on connect).
    """
    match = DATA_PATTERN.search(line)

    if not match:
        return None

    try:
        values = [float(g) for g in match.groups()]
    except ValueError:
        return None

    if not all(math.isfinite(v) for v in values):
        return None

    return {
        "foot_roll": values[0],
        "foot_pitch": values[1],
        "foot_yaw": values[2],
        "shank_roll": values[3],
        "shank_pitch": values[4],
        "shank_yaw": values[5],
        "fw_ankle": values[6],
    }


# ============================================================
# QUATERNION UTILITIES  (unchanged)
#
# The firmware only transmits Euler angles, but Euler -> quaternion
# is invertible (away from gimbal lock at |pitch| = 90 deg), so the
# foot and shank orientations can be rebuilt here. This is what makes
# inversion / eversion possible without touching the firmware.
# ============================================================

def euler_to_quaternion(roll_deg, pitch_deg, yaw_deg):
    """
    Inverse of the firmware's getRoll / getPitch / getYaw.
    Standard ZYX (aerospace) convention.
    """
    r = math.radians(roll_deg) * 0.5
    p = math.radians(pitch_deg) * 0.5
    y = math.radians(yaw_deg) * 0.5

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quaternion_normalize(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)

    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])

    return q / norm


def quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quaternion_multiply(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quaternion_to_euler(q):
    """Returns roll, pitch, yaw in degrees. Mirrors the firmware."""
    w, x, y, z = quaternion_normalize(q)

    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)

    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))


def relative_quaternion(sample):
    """
    Rebuilds conj(qShank) * qFoot from the transmitted Euler angles.
    Same expression the firmware uses.
    """
    q_foot = euler_to_quaternion(
        sample["foot_roll"], sample["foot_pitch"], sample["foot_yaw"]
    )
    q_shank = euler_to_quaternion(
        sample["shank_roll"], sample["shank_pitch"], sample["shank_yaw"]
    )

    return quaternion_normalize(
        quaternion_multiply(quaternion_conjugate(q_shank), q_foot)
    )


# ============================================================
# EXERCISE CONFIGURATION  (targets unchanged)
#
# axis   : which component of the neutral-corrected relative
#          orientation carries this movement.
#          "firmware" = use the Ankle value the ESP32 already sends
#          (this is roll of qMotion, validated against Kinovea).
# sign   : +1 or -1, so the movement always reads positive.
#          VERIFY THIS ON YOUR OWN HARDWARE - see the polarity
#          check on the Live Session page (Developer mode).
# ============================================================

EXERCISES = {
    "Dorsiflexion": {
        "description": "Move the foot upward toward the shin.",
        "axis": "firmware",
        "sign": -1,
        "target_min": 15,
        "target_max": 25,
        "target_reps": 15,
    },
    "Plantarflexion": {
        "description": "Point the foot downward away from the shin.",
        "axis": "firmware",
        "sign": 1,
        "target_min": 20,
        "target_max": 50,
        "target_reps": 15,
    },
    "Inversion": {
        "description": "Turn the sole of the foot inward.",
        "axis": "pitch",
        "sign": 1,
        "target_min": 10,
        "target_max": 35,
        "target_reps": 15,
    },
    "Eversion": {
        "description": "Turn the sole of the foot outward.",
        "axis": "pitch",
        "sign": -1,
        "target_min": 10,
        "target_max": 20,
        "target_reps": 15,
    },
}

AXIS_CHOICES = ["firmware", "roll", "pitch", "yaw"]


# ============================================================
# DATABASE
#
# `sessions`     : one summary row per session  (UNCHANGED)
# `repetitions`  : one row per attempt          (NEW, additive)
# ============================================================

SESSION_COLUMNS = [
    ("patient_id", "TEXT"),
    ("date", "TEXT"),
    ("exercise", "TEXT"),
    ("axis", "TEXT"),
    ("sign", "INTEGER"),
    ("target_min", "REAL"),
    ("target_max", "REAL"),
    ("achieved_rom", "REAL"),
    ("peak_angle", "REAL"),
    ("mean_peak", "REAL"),
    ("repetitions", "INTEGER"),
    ("correct_repetitions", "INTEGER"),
    ("incorrect_repetitions", "INTEGER"),
    ("score", "REAL"),
    ("duration", "REAL"),
    ("samples", "INTEGER"),
    ("source", "TEXT"),
    ("notes", "TEXT"),
]

# NEW: per-repetition table.
# rep_number        : 1, 2, 3 ... attempt order inside the session
# peak_angle        : highest angle reached during that attempt
# duration_s        : time from leaving neutral to returning to neutral
# result            : "correct" or "incorrect"
# reason            : "ok", "too_short" or "too_far"
# target_min / max  : stored per row so each rep is self-describing
REPETITION_COLUMNS = [
    ("session_id", "INTEGER"),
    ("patient_id", "TEXT"),
    ("exercise", "TEXT"),
    ("rep_number", "INTEGER"),
    ("peak_angle", "REAL"),
    ("duration_s", "REAL"),
    ("result", "TEXT"),
    ("reason", "TEXT"),
    ("target_min", "REAL"),
    ("target_max", "REAL"),
]


def initialize_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            sex TEXT,
            created_at TEXT
        )
    """)

    # Migrate an older database that predates the "sex" column.
    cursor.execute("PRAGMA table_info(patients)")
    existing_patient_cols = {row[1] for row in cursor.fetchall()}

    if "sex" not in existing_patient_cols:
        cursor.execute("ALTER TABLE patients ADD COLUMN sex TEXT")

    column_sql = ",\n".join(f"{name} {ctype}" for name, ctype in SESSION_COLUMNS)

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {column_sql}
        )
    """)

    # Migrate an older database created by v1 of this app
    cursor.execute("PRAGMA table_info(sessions)")
    existing = {row[1] for row in cursor.fetchall()}

    for name, ctype in SESSION_COLUMNS:
        if name not in existing:
            cursor.execute(f"ALTER TABLE sessions ADD COLUMN {name} {ctype}")

    # NEW: repetitions table. CREATE IF NOT EXISTS means an existing
    # database simply gains the table; nothing already stored changes.
    rep_column_sql = ",\n".join(f"{name} {ctype}" for name, ctype in REPETITION_COLUMNS)

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS repetitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {rep_column_sql}
        )
    """)

    conn.commit()
    conn.close()


initialize_database()


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "page": "Login",
    "logged_in": False,

    # NEW: "Patient" hides technical panels, "Developer" shows them.
    "mode": "Patient",

    "patient_id": "",
    "patient_name": "",
    "patient_age": 0,
    "patient_sex": "",

    "exercise": "Dorsiflexion",
    "axis_overrides": {},
    "sign_overrides": {},

    "entry_fraction": 0.4,
    "return_threshold": 5.0,
    "min_peak_attempt": DEFAULT_MIN_PEAK_ATTEMPT,

    # Serial
    "serial_object": None,
    "serial_connected": False,
    "serial_port": None,
    "rx_buffer": "",
    "log_lines": [],
    "device_ready": False,
    "connect_time": None,
    "lines_received": 0,

    # Calibration
    "calibrated": False,
    "awaiting_neutral": False,
    "neutral_quaternion": None,

    # Acquisition
    "simulation": False,
    "sim_index": 0,
    "running": False,
    "session_complete": False,
    "sample_count": 0,

    "timestamps": [],
    "angles": [],
    "raw_fw_angles": [],
    "velocities": [],
    "rel_roll": [],
    "rel_pitch": [],
    "rel_yaw": [],
    "foot_rpy": [],
    "shank_rpy": [],

    "current_angle": 0.0,
    "current_velocity": 0.0,
    "current_sample": None,

    "min_angle": None,
    "max_angle": None,

    "movement_state": "NEUTRAL",
    "peak_angle": 0.0,
    "rep_start_index": 0,
    "rep_peaks": [],

    # NEW: full record of each attempt, written to SQLite at save time.
    "rep_records": [],

    "repetitions": 0,
    "correct_repetitions": 0,
    "incorrect_repetitions": 0,

    # Feedback shown right after the most recent completed rep.
    "last_rep_correct": None,
    "last_rep_peak": None,
    "last_rep_reason": "",

    # Set the instant target_reps correct repetitions are reached.
    "goal_reached": False,
    "session_saved": False,

    # NEW: makes sure the completion effect and quote appear exactly once.
    "celebration_shown": False,
    "completion_quote": "",
    "session_notes": "",
    "last_saved_session_id": None,

    # NEW: the quote shown on the Exercise page.
    "exercise_quote": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


st.set_page_config(
    page_title="Ankle Rehabilitation System",
    layout="wide",
)


# ============================================================
# STYLING  (v4.0)
#
# Two completely separate themes, so switching mode never looks like the
# same screen with extra widgets added:
#
#   PATIENT_THEME    warm, rounded, colourful. Built to keep a person
#                    motivated: progress ring, streak, badges, big
#                    coloured coaching text.
#   DEVELOPER_THEME  dark, dense, monospace numbers. A workbench.
#
# Colours (patient):
#   teal  #0F766E   calm / trust          coral #FB7185  energy
#   gold  #F59E0B   achievement           mint  #ECFDF5  background
#   ink   #0F172A   text
#
# Motion is only used where it answers something the patient did
# (a rep landing, the goal being reached) and is switched off entirely
# for anyone with "reduce motion" set in their operating system.
# ============================================================

PATIENT_THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

/* ---------- force a consistent LIGHT look regardless of the browser's
   or OS's dark-mode setting -----------------------------------------
   Streamlit auto-switches its own base text colour to a light grey
   when the visitor's system is in dark mode. That collided with this
   theme's light cards and produced pale/invisible text. The rules
   below pin every common widget to explicit colours so the page reads
   the same in light or dark mode. */
:root { color-scheme: light; }

html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {
    font-family: 'Nunito', 'Segoe UI', system-ui, sans-serif !important;
}
.stApp {
    background:
        radial-gradient(1100px 550px at 92% -12%, #ffe1ea 0%, rgba(255,225,234,0) 62%),
        radial-gradient(950px 550px at -10% -5%, #cdf7ef 0%, rgba(205,247,239,0) 62%),
        radial-gradient(800px 500px at 50% 115%, #fef3c7 0%, rgba(254,243,199,0) 60%),
        #f4fbf8;
}
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
label, .stCaption, p, span, li, small {
    color: #0f172a;
}
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label { color: #0f172a !important; font-weight: 700; }
[data-testid="stAlert"] * { color: #0f172a !important; }
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    color: #0f172a !important; background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #94a3b8 !important; }
[data-baseweb="select"] * , [data-baseweb="popover"] * , [data-baseweb="menu"] * {
    color: #0f172a !important;
}
[data-baseweb="select"] > div { background: #ffffff !important; border-color: #cbd5e1 !important; }
[data-testid="stDataFrame"] * { color: #0f172a !important; }
[data-testid="stMetricValue"] { color: #0f172a !important; }
[data-testid="stMetricLabel"] { color: #475569 !important; }
.stRadio label, .stCheckbox label, .stRadio p, .stCheckbox p { color: #0f172a !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] p { color: #0f172a !important; }
.stTabs [data-baseweb="tab-list"] {
    flex-wrap: wrap !important; gap: .35rem !important;
    border-bottom: 1px solid #cbd5e1 !important;
}
.stTabs [data-baseweb="tab"] {
    white-space: normal !important; min-width: max-content !important;
    background: #e2e8f0 !important; border-radius: 8px 8px 0 0 !important;
    border: 1px solid #cbd5e1 !important; padding: .45rem .9rem !important;
}
.stTabs [data-baseweb="tab"] p { color: #334155 !important; font-weight: 700 !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #0f766e !important; border-color: #0f766e !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] p { color: #ffffff !important; }
.stTabs [data-baseweb="tab"]:focus { outline: 2px solid #14b8a6 !important; }
a { color: #0d9488 !important; font-weight: 700; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f766e 0%, #115e59 100%);
}
[data-testid="stSidebar"] * { color: #ecfdf5 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.25);
    border-radius: 12px;
    font-weight: 700;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,.24);
}

/* Sidebar navigation contrast fix: keep inactive labels/icons readable. */
[data-testid="stSidebar"] .stButton > button[kind="secondary"],
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:enabled {
    background: rgba(7, 55, 52, .42) !important;
    color: #ffffff !important;
    border: 1px solid rgba(204, 251, 241, .65) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] *,
[data-testid="stSidebar"] .stButton > button[kind="secondary"] span,
[data-testid="stSidebar"] .stButton > button[kind="secondary"] p {
    color: #ffffff !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background: rgba(7, 55, 52, .72) !important;
    border-color: #ccfbf1 !important;
}
[data-testid="stSidebar"] .stButton > button:disabled,
[data-testid="stSidebar"] .stButton > button:disabled * {
    color: #ffffff !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[kind="primary"] * {
    color: #ffffff !important;
    opacity: 1 !important;
}
h1, h2, h3 { color: #0f172a; font-weight: 800 !important; letter-spacing: -.01em; }

/* ---------- hero ---------- */
.p-hero {
    display: flex; align-items: center; gap: 1.6rem; flex-wrap: wrap;
    background: linear-gradient(120deg, #0f766e 0%, #0d9488 50%, #14b8a6 100%);
    color: #fff; border-radius: 22px; padding: 1.5rem 1.8rem; margin-bottom: 1.1rem;
    box-shadow: 0 10px 28px rgba(15,118,110,.25);
}
.p-hero .txt { flex: 1 1 260px; }
.p-hero h2 { color: #fff; margin: 0 0 .3rem 0; font-size: 1.9rem; }
.p-hero p  { margin: 0; font-size: 1.05rem; opacity: .95; }

/* ---------- progress ring (pure CSS conic-gradient) ---------- */
.ring {
    --p: 0; --c: #fbbf24;
    width: 132px; height: 132px; border-radius: 50%;
    background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,.28) 0);
    display: grid; place-items: center; flex: 0 0 auto;
}
.ring::before {
    content: ""; width: 102px; height: 102px; border-radius: 50%;
    background: #0f766e; grid-area: 1 / 1;
}
.ring span {
    grid-area: 1 / 1; z-index: 1; font-weight: 900; font-size: 1.6rem; color: #fff;
    text-align: center; line-height: 1.05;
}
.ring small { display: block; font-size: .72rem; font-weight: 700; opacity: .85; }

/* ---------- streak + stat tiles ---------- */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .8rem; margin: .4rem 0 1.1rem 0; }
.tile {
    background: #fff; border-radius: 18px; padding: 1rem 1.1rem;
    box-shadow: 0 4px 14px rgba(15,23,42,.07); border-top: 5px solid var(--a, #14b8a6);
    transition: transform .15s ease, box-shadow .15s ease;
}
.tile:hover { transform: translateY(-3px); box-shadow: 0 10px 22px rgba(15,23,42,.14); }
.tile .v { font-size: 1.9rem; font-weight: 900; color: #0f172a; line-height: 1.1; }
.tile .l { font-size: .85rem; font-weight: 700; color: #475569; margin-top: .15rem; }
.tile.gold  { --a: #f59e0b; }
.tile.coral { --a: #fb7185; }
.tile.teal  { --a: #14b8a6; }
.tile.blue  { --a: #3b82f6; }

/* ---------- exercise cards ---------- */
.ex-card {
    border-radius: 20px; padding: 1.1rem 1.2rem; color: #fff; height: 172px;
    box-shadow: 0 8px 20px rgba(15,23,42,.14); position: relative; overflow: hidden;
    margin-bottom: .55rem; display: flex; flex-direction: column;
    transition: transform .15s ease, box-shadow .15s ease;
}
.ex-card:hover { transform: translateY(-4px) scale(1.01); box-shadow: 0 14px 28px rgba(15,23,42,.22); }
.ex-card h4 { margin: 0 0 .25rem 0; color: #fff; font-size: 1.2rem; font-weight: 800; }
.ex-card p  { margin: 0; font-size: .92rem; opacity: .95; padding-right: 3.4rem; flex: 1 0 auto; }
.ex-card .emoji { position: absolute; right: 12px; top: 10px; font-size: 2.4rem; opacity: .35; }
.ex-card .tag {
    display: inline-block; margin-top: .5rem; background: rgba(255,255,255,.25);
    padding: .15rem .65rem; border-radius: 999px; font-size: .78rem; font-weight: 800;
    align-self: flex-start;
}
.ex-dorsi  { background: linear-gradient(135deg, #0ea5e9, #2563eb); }
.ex-plant  { background: linear-gradient(135deg, #f97316, #ef4444); }
.ex-inver  { background: linear-gradient(135deg, #8b5cf6, #d946ef); }
.ex-ever   { background: linear-gradient(135deg, #10b981, #0d9488); }
.ex-card.chosen { outline: 4px solid #fbbf24; }

/* ---------- badges ---------- */
.badges { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .7rem; margin: .3rem 0 1rem 0; }
.badge {
    background: #fff; border-radius: 16px; padding: .8rem .9rem;
    box-shadow: 0 4px 12px rgba(15,23,42,.07); text-align: center;
}
.badge .ic { font-size: 1.8rem; }
.badge .nm { font-weight: 800; color: #0f172a; font-size: .92rem; }
.badge .ds { font-size: .75rem; color: #64748b; font-weight: 600; }
.badge.locked { filter: grayscale(1); opacity: .5; }

/* ---------- quote ---------- */
.quote-card {
    background: linear-gradient(120deg, #fff7ed, #ffedd5);
    border-left: 7px solid #fb923c; padding: 1rem 1.3rem; border-radius: 16px;
    color: #7c2d12; font-size: 1.12rem; font-weight: 700; margin: .7rem 0 1rem 0;
}

/* ---------- live coaching ---------- */
.instruction-card {
    padding: 1.4rem 1.4rem; border-radius: 22px; font-size: 1.7rem; font-weight: 900;
    text-align: center; margin: .5rem 0 .9rem 0; box-shadow: 0 8px 22px rgba(15,23,42,.10);
}
.instruction-neutral { background: linear-gradient(120deg, #e0f2fe, #bae6fd); color: #075985; }
.instruction-more    { background: linear-gradient(120deg, #fef3c7, #fde68a); color: #92400e; }
.instruction-good    { background: linear-gradient(120deg, #bbf7d0, #86efac); color: #14532d; }
.instruction-back    { background: linear-gradient(120deg, #fecaca, #fca5a5); color: #7f1d1d; }

/* ---------- rep dots ---------- */
.dots { display: flex; gap: .45rem; flex-wrap: wrap; justify-content: center; margin: .6rem 0 .3rem 0; }
.dot {
    width: 34px; height: 34px; border-radius: 50%; display: grid; place-items: center;
    font-weight: 900; font-size: .85rem; background: #e2e8f0; color: #94a3b8;
}
.dot.done { background: linear-gradient(135deg, #22c55e, #14b8a6); color: #fff; animation: pop .35s ease-out; }
.dot.next { outline: 3px solid #fb7185; color: #fb7185; background: #fff; }
.dots-cap { text-align: center; font-weight: 800; color: #0f766e; margin-bottom: .6rem; }

/* ---------- angle gauge ---------- */
.gauge-wrap { background: #fff; border-radius: 22px; padding: 1.1rem 1.3rem; box-shadow: 0 6px 18px rgba(15,23,42,.08); }
.gauge-num  { font-size: 3.6rem; font-weight: 900; color: #0f172a; line-height: 1; }
.gauge-num small { font-size: 1.1rem; color: #64748b; font-weight: 800; }
.gauge-track {
    position: relative; height: 26px; border-radius: 999px; background: #e2e8f0; margin-top: .9rem; overflow: hidden;
}
.gauge-target { position: absolute; top: 0; bottom: 0; background: rgba(34,197,94,.35); border-left: 3px solid #16a34a; border-right: 3px solid #16a34a; }
.gauge-fill   { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 999px; background: linear-gradient(90deg, #38bdf8, #6366f1); transition: width .12s linear; }
.gauge-scale  { display: flex; justify-content: space-between; font-size: .75rem; font-weight: 800; color: #64748b; margin-top: .25rem; }

/* ---------- completion ---------- */
.complete-card {
    background: linear-gradient(135deg, #16a34a 0%, #0d9488 60%, #0ea5e9 100%);
    color: #fff; padding: 1.8rem 2rem; border-radius: 24px; margin: 1rem 0;
    box-shadow: 0 12px 30px rgba(13,148,136,.32); animation: pop .6s ease-out;
}
.complete-card h2 { margin: 0 0 .4rem 0; color: #fff; font-size: 2.1rem; }
.complete-card p  { margin: .25rem 0; font-size: 1.1rem; }
/* ---------- session cards (My progress) ---------- */
.s-card {
    background: #fff; border-radius: 16px; padding: .85rem 1.1rem; margin-bottom: .6rem;
    box-shadow: 0 4px 12px rgba(15,23,42,.07); border-left: 6px solid #14b8a6;
}
.s-top   { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; flex-wrap: wrap; }
.s-name  { font-weight: 800; color: #0f172a; font-size: 1.05rem; }
.s-when  { font-weight: 700; color: #64748b; font-size: .85rem; }
.s-stats { display: flex; gap: 1.1rem; flex-wrap: wrap; margin-top: .35rem; font-weight: 700; color: #334155; font-size: .9rem; }
.s-note  { margin-top: .5rem; background: #fff7ed; color: #7c2d12; border-radius: 10px; padding: .4rem .7rem; font-weight: 600; font-size: .9rem; }

.stat-pill {
    display: inline-block; background: rgba(255,255,255,.24); padding: .3rem .9rem;
    border-radius: 999px; margin: .6rem .45rem 0 0; font-weight: 800;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, #f43f5e, #fb7185); border: 0; color: #fff;
    border-radius: 14px; font-weight: 800; padding: .6rem 1rem;
    box-shadow: 0 6px 16px rgba(244,63,94,.35);
}
.stButton > button {
    border-radius: 14px; font-weight: 700; min-height: 2.6rem;
    transition: transform .12s ease, box-shadow .12s ease;
}
.stButton > button:hover { transform: translateY(-1px); }
.stButton > button[kind="secondary"] {
    background: #fff; border: 2px solid #0d9488; color: #0d9488 !important;
}
.stButton > button[kind="secondary"]:hover { background: #ecfdf5; }
.stButton > button:disabled,
.stButton > button[kind="primary"]:disabled {
    background: #e2e8f0 !important; color: #94a3b8 !important;
    box-shadow: none !important; cursor: not-allowed;
}

@keyframes pop {
    0%   { transform: scale(.9); opacity: 0; }
    100% { transform: scale(1);  opacity: 1; }
}
@media (prefers-reduced-motion: reduce) {
    .complete-card, .dot.done { animation: none; }
    .gauge-fill { transition: none; }
}
</style>
"""

DEVELOPER_THEME = """
<style>
/* Pin an explicit dark look regardless of the browser's or OS's
   light/dark setting, so labels, inputs and dropdowns never inherit a
   pale-on-pale or dark-on-dark combination from Streamlit's own theme. */
:root { color-scheme: dark; }

html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}
.stApp { background: #0b1220; color: #cbd5e1; }
[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #1e293b; border: 1px solid #334155; border-radius: 6px; font-weight: 600;
}
[data-testid="stSidebar"] .stButton > button:hover { border-color: #38bdf8; }
h1, h2, h3, h4 { color: #e2e8f0; font-weight: 700 !important; }
.stMarkdown, .stCaption, label, p, span, div, li, small { color: #cbd5e1; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: #e2e8f0 !important; }
[data-testid="stAlert"] * { color: #e2e8f0 !important; }
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    color: #e2e8f0 !important; background: #111a2e !important; border: 1px solid #334155 !important;
}
[data-baseweb="select"] * , [data-baseweb="popover"] * , [data-baseweb="menu"] * {
    color: #e2e8f0 !important; background: #111a2e !important;
}
[data-testid="stDataFrame"] * { color: #e2e8f0 !important; }
.stRadio label, .stCheckbox label, .stRadio p, .stCheckbox p { color: #e2e8f0 !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] p { color: #e2e8f0 !important; }
.stTabs [data-baseweb="tab-list"] {
    flex-wrap: wrap !important; gap: .35rem !important;
    border-bottom: 1px solid #334155 !important;
}
.stTabs [data-baseweb="tab"] {
    white-space: normal !important; min-width: max-content !important;
    background: #1e293b !important; border-radius: 8px 8px 0 0 !important;
    border: 1px solid #334155 !important; padding: .45rem .9rem !important;
}
.stTabs [data-baseweb="tab"] p { color: #cbd5e1 !important; font-weight: 700 !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #0284c7 !important; border-color: #38bdf8 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] p { color: #ffffff !important; }
.stTabs [data-baseweb="tab"]:focus { outline: 2px solid #38bdf8 !important; }
a { color: #38bdf8 !important; }
[data-testid="stMetric"] {
    background: #111a2e; border: 1px solid #1e293b; border-radius: 8px; padding: .7rem .9rem;
}
[data-testid="stMetricValue"] { font-family: 'Consolas', 'Courier New', monospace !important; color: #38bdf8; }
[data-testid="stMetricLabel"] { color: #94a3b8; }
.dev-banner {
    background: #111a2e; border: 1px solid #1e293b; border-left: 4px solid #38bdf8;
    border-radius: 6px; padding: .6rem 1rem; margin-bottom: 1rem; color: #cbd5e1;
    font-family: 'Consolas', monospace; font-size: .9rem;
}
.dev-banner b { color: #38bdf8; }
.stButton > button { border-radius: 6px; font-weight: 600; }
.stButton > button[kind="primary"] { background: #0284c7; border: 2px solid #7dd3fc; color: #fff; font-weight: 800; box-shadow: 0 0 0 2px rgba(56,189,248,.18); }
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #f0fdfa !important; color: #115e59 !important;
    border: 2px solid #5eead4 !important; font-weight: 800 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background: rgba(255,255,255,.10) !important; color: #ecfdf5 !important;
    border: 1px solid rgba(255,255,255,.28) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,.20) !important; border-color: #99f6e4 !important;
}
pre, code { font-family: 'Consolas', 'Courier New', monospace !important; }

/* completion_panel is shared with the patient view; these give it a
   plain, technical look here instead of leaving it unstyled. */
.complete-card {
    background: #111a2e; border: 1px solid #1e293b; border-left: 4px solid #22c55e;
    border-radius: 8px; padding: 1.2rem 1.4rem; margin: 1rem 0; color: #cbd5e1;
}
.complete-card h2 { color: #4ade80; margin: 0 0 .4rem 0; font-size: 1.3rem; }
.complete-card p { margin: .2rem 0; color: #cbd5e1; }
.stat-pill {
    display: inline-block; background: #1e293b; border: 1px solid #334155;
    padding: .25rem .8rem; border-radius: 4px; margin: .5rem .4rem 0 0;
    font-family: 'Consolas', monospace; font-size: .85rem; color: #38bdf8;
}
.quote-card {
    background: #111a2e; border-left: 4px solid #64748b; border-radius: 6px;
    padding: .7rem 1rem; margin: .6rem 0; color: #94a3b8; font-style: italic;
}
</style>
"""


def inject_style():
    """Applies the theme that matches the current mode. Called on every page."""
    if st.session_state.mode == "Developer":
        st.markdown(DEVELOPER_THEME, unsafe_allow_html=True)
    else:
        st.markdown(PATIENT_THEME, unsafe_allow_html=True)


# ============================================================
# CONFIG ACCESS (with per-session overrides from Settings)
# ============================================================

def exercise_config(name=None):
    name = name or st.session_state.exercise
    config = dict(EXERCISES[name])

    if name in st.session_state.axis_overrides:
        config["axis"] = st.session_state.axis_overrides[name]

    if name in st.session_state.sign_overrides:
        config["sign"] = st.session_state.sign_overrides[name]

    return config


def is_developer():
    return st.session_state.mode == "Developer"


# ============================================================
# SESSION RESET
# ============================================================

def reset_measurements():
    st.session_state.running = False
    st.session_state.session_complete = False
    st.session_state.sample_count = 0

    st.session_state.timestamps = []
    st.session_state.angles = []
    st.session_state.raw_fw_angles = []
    st.session_state.velocities = []
    st.session_state.rel_roll = []
    st.session_state.rel_pitch = []
    st.session_state.rel_yaw = []
    st.session_state.foot_rpy = []
    st.session_state.shank_rpy = []

    st.session_state.current_angle = 0.0
    st.session_state.current_velocity = 0.0

    st.session_state.min_angle = None
    st.session_state.max_angle = None

    st.session_state.movement_state = "NEUTRAL"
    st.session_state.peak_angle = 0.0
    st.session_state.rep_start_index = 0
    st.session_state.rep_peaks = []
    st.session_state.rep_records = []

    st.session_state.repetitions = 0
    st.session_state.correct_repetitions = 0
    st.session_state.incorrect_repetitions = 0

    st.session_state.last_rep_correct = None
    st.session_state.last_rep_peak = None
    st.session_state.last_rep_reason = ""
    st.session_state.goal_reached = False
    st.session_state.session_saved = False

    st.session_state.celebration_shown = False
    st.session_state.completion_quote = ""
    st.session_state.session_notes = ""
    st.session_state.last_saved_session_id = None

    # Force the cached rep-history image to be rebuilt for the new session.
    st.session_state.rep_chart_key = None
    st.session_state.rep_chart_png = None

    st.session_state.sim_index = 0


def log(message):
    st.session_state.log_lines.append(
        f"{datetime.now().strftime('%H:%M:%S')}  {message}"
    )
    st.session_state.log_lines = st.session_state.log_lines[-300:]


# ============================================================
# SERIAL LAYER  (unchanged)
# ============================================================

def available_ports():
    if not SERIAL_AVAILABLE:
        return []

    try:
        return [p.device for p in serial.tools.list_ports.comports()]
    except Exception:
        return []


def disconnect_serial():
    ser = st.session_state.serial_object

    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass

    st.session_state.serial_object = None
    st.session_state.serial_connected = False
    st.session_state.serial_port = None
    st.session_state.rx_buffer = ""
    st.session_state.device_ready = False
    st.session_state.connect_time = None
    st.session_state.calibrated = False
    st.session_state.awaiting_neutral = False
    st.session_state.neutral_quaternion = None


def connect_serial(port, baudrate):
    if not SERIAL_AVAILABLE:
        st.error("pyserial is not installed. Run: pip install pyserial")
        return False

    # Always release any stale handle first. Streamlit hot-reload leaves
    # the port locked otherwise.
    disconnect_serial()

    try:
        ser = serial.Serial(port, baudrate, timeout=0)
        time.sleep(0.2)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception as exc:
        st.error(f"Could not open {port}: {exc}")
        return False

    st.session_state.serial_object = ser
    st.session_state.serial_connected = True
    st.session_state.serial_port = port
    st.session_state.rx_buffer = ""
    st.session_state.connect_time = time.time()
    st.session_state.device_ready = False
    st.session_state.lines_received = 0

    log(f"Opened {port} at {baudrate} baud. ESP32 will reboot and "
        f"run gyro calibration - keep both IMUs still.")

    return True


def read_serial_lines():
    """
    Reads every byte waiting on the port and splits it into complete
    lines. A partial trailing line is kept in rx_buffer for next time.

    This must drain the buffer, not read a single line. The firmware
    sends 100 lines/second, the Streamlit loop wakes ~20 times/second,
    so single-line reads would fall further behind real time every
    second.
    """
    ser = st.session_state.serial_object

    if ser is None:
        return []

    try:
        waiting = ser.in_waiting
        if waiting:
            chunk = ser.read(waiting).decode("utf-8", errors="ignore")
            st.session_state.rx_buffer += chunk
    except Exception as exc:
        log(f"Serial read failed: {exc}")
        disconnect_serial()
        return []

    if "\n" not in st.session_state.rx_buffer:
        return []

    parts = st.session_state.rx_buffer.split("\n")
    st.session_state.rx_buffer = parts[-1]

    return [p.strip() for p in parts[:-1] if p.strip()]


def send_neutral_command():
    ser = st.session_state.serial_object

    if ser is None:
        st.warning("Not connected.")
        return

    try:
        ser.write(b"N")
        st.session_state.awaiting_neutral = True
        log("Sent 'N'. Firmware pauses ~2 s while it captures neutral.")
    except Exception as exc:
        st.error(f"Could not send command: {exc}")


# ============================================================
# SIMULATED FIRMWARE OUTPUT  (unchanged, see note below)
#
# Generates the same text format the ESP32 produces, so the parser
# and the whole downstream pipeline are exercised identically in
# simulation and on real hardware.
#
# The simulated wave always peaks at the middle of the target band,
# so simulation mode produces only correct repetitions. To see
# incorrect repetitions being recorded, use the Developer mode
# "Simulated movement quality" control on the Live page.
# ============================================================

def simulated_lines(count):
    config = exercise_config()
    lines = []

    quality = st.session_state.get("sim_quality", "Correct")
    mid = (config["target_min"] + config["target_max"]) / 2.0

    def peak_for_cycle(cycle):
        """
        Amplitude for one whole movement cycle. It is decided from the
        cycle number so the amplitude never changes part-way through a
        repetition, which would create a half-sized fake movement.
        """
        if quality == "Too short":
            return max(st.session_state.min_peak_attempt + 1.0,
                       config["target_min"] * 0.5)
        if quality == "Too far":
            return config["target_max"] * 1.4
        if quality == "Mixed":
            # Every third cycle is short: correct, correct, short, ...
            if cycle % 3 == 2:
                return config["target_min"] * 0.5
            return mid
        return mid

    # One full movement is one |sin| hump = 2 s at 0.25 Hz of the sine.
    # abs(sin) has period 1/(2*0.25) = 2 s.
    cycle_samples = int(round(2.0 * FIRMWARE_RATE_HZ))

    for _ in range(count):
        index = st.session_state.sim_index
        t = index * SAMPLE_PERIOD
        st.session_state.sim_index += 1

        peak = peak_for_cycle(index // cycle_samples)

        wave = peak * abs(math.sin(2 * math.pi * 0.25 * t))

        # Produce the movement on the axis this exercise expects
        signed = config["sign"] * wave

        if config["axis"] in ("firmware", "roll"):
            foot = (signed, 0.0, 0.0)
        elif config["axis"] == "pitch":
            foot = (0.0, signed, 0.0)
        else:
            foot = (0.0, 0.0, signed)

        ankle = signed if config["axis"] in ("firmware", "roll") else 0.0

        lines.append(
            f"Foot R/P/Y: {foot[0]:.1f} / {foot[1]:.1f} / {foot[2]:.1f}"
            f"   |   Shank R/P/Y: 0.0 / 0.0 / 0.0"
            f"   |   Ankle: {ankle:.2f} deg"
        )

    return lines


# ============================================================
# SAMPLE PROCESSING
# ============================================================

def movement_angle(sample, q_motion):
    """
    Returns the signed movement angle for the active exercise.

    For dorsi/plantarflexion the firmware's own Ankle value is used,
    because that is the number validated against Kinovea. For
    inversion/eversion the axis is taken from the reconstructed
    relative orientation.
    """
    config = exercise_config()

    if config["axis"] == "firmware":
        raw = sample["fw_ankle"]
    else:
        roll, pitch, yaw = quaternion_to_euler(q_motion)
        raw = {"roll": roll, "pitch": pitch, "yaw": yaw}[config["axis"]]

    return config["sign"] * raw, raw


def finish_goal():
    """Called once, the instant the correct-rep goal is reached."""
    st.session_state.goal_reached = True
    st.session_state.running = False
    st.session_state.session_complete = True


def update_repetition(signed_angle):
    """
    Two-state counter with peak tracking.

    A repetition is registered on the return to neutral, and it is
    graded on the PEAK reached during that repetition - not on the
    angle at the moment of return, which is always near zero.

    CHANGED in v3.0:
      * The entry threshold is now low (a fraction of MIN_PEAK, not of
        the target) so small movements begin an attempt.
      * When the patient returns to neutral, the attempt is recorded if
        its peak reached min_peak_attempt. Smaller peaks are treated as
        noise and discarded.
      * A recorded attempt is graded:
            peak < target_min   -> incorrect, reason "too_short"
            peak > target_max   -> incorrect, reason "too_far"
            otherwise           -> correct,   reason "ok"

    UNCHANGED:
      * Only CORRECT repetitions count toward the session goal.
      * Once target_reps correct reps are reached, the session stops.
    """
    config = st.session_state
    exercise = exercise_config()

    target_min = exercise["target_min"]
    target_max = exercise["target_max"]
    target_reps = exercise["target_reps"]

    # Once the goal is already met, stop counting further attempts.
    if st.session_state.correct_repetitions >= target_reps:
        if not st.session_state.goal_reached:
            finish_goal()
        return

    release = st.session_state.return_threshold

    # A movement must rise above the return threshold to begin an
    # attempt, otherwise the state would flip back instantly.
    entry = release + 1.0

    if st.session_state.movement_state == "NEUTRAL":
        if signed_angle >= entry:
            st.session_state.movement_state = "MOVING"
            st.session_state.peak_angle = signed_angle
            st.session_state.rep_start_index = st.session_state.sample_count

    elif st.session_state.movement_state == "MOVING":
        if signed_angle > st.session_state.peak_angle:
            st.session_state.peak_angle = signed_angle

        if signed_angle <= release:
            peak = st.session_state.peak_angle
            samples_taken = st.session_state.sample_count - st.session_state.rep_start_index
            duration = samples_taken * SAMPLE_PERIOD

            # Return to neutral either way.
            st.session_state.movement_state = "NEUTRAL"
            st.session_state.peak_angle = 0.0

            # Below the noise floor: not a real attempt, ignore it.
            if peak < st.session_state.min_peak_attempt:
                return

            if peak < target_min:
                result, reason = "incorrect", "too_short"
            elif peak > target_max:
                result, reason = "incorrect", "too_far"
            else:
                result, reason = "correct", "ok"

            st.session_state.repetitions += 1
            st.session_state.rep_peaks.append(round(peak, 2))

            st.session_state.rep_records.append({
                "rep_number": st.session_state.repetitions,
                "peak_angle": round(peak, 2),
                "duration_s": round(duration, 2),
                "result": result,
                "reason": reason,
                "target_min": target_min,
                "target_max": target_max,
            })

            st.session_state.last_rep_peak = round(peak, 2)
            st.session_state.last_rep_correct = (result == "correct")

            if result == "correct":
                st.session_state.correct_repetitions += 1
                st.session_state.last_rep_reason = "Good range of motion."
            else:
                st.session_state.incorrect_repetitions += 1
                if reason == "too_short":
                    st.session_state.last_rep_reason = (
                        f"Too short - reached {peak:.1f} deg, "
                        f"need at least {target_min:.0f} deg."
                    )
                else:
                    st.session_state.last_rep_reason = (
                        f"Too far - reached {peak:.1f} deg, "
                        f"stay under {target_max:.0f} deg."
                    )

            # Stop the instant the correct-rep goal is hit.
            if st.session_state.correct_repetitions >= target_reps:
                finish_goal()


def process_sample(sample):
    q_relative = relative_quaternion(sample)

    # Neutral capture on the GUI side, taken from the first sample
    # after the firmware confirms it saved its own neutral.
    if st.session_state.awaiting_neutral:
        st.session_state.neutral_quaternion = q_relative
        st.session_state.awaiting_neutral = False
        st.session_state.calibrated = True
        log("Neutral captured on GUI side.")

    if st.session_state.neutral_quaternion is not None:
        q_motion = quaternion_normalize(
            quaternion_multiply(
                quaternion_conjugate(st.session_state.neutral_quaternion),
                q_relative,
            )
        )
    else:
        q_motion = q_relative

    signed, raw = movement_angle(sample, q_motion)

    st.session_state.current_sample = sample
    st.session_state.current_angle = signed

    if not st.session_state.running:
        return

    index = st.session_state.sample_count
    elapsed = index * SAMPLE_PERIOD
    st.session_state.sample_count += 1

    if st.session_state.angles:
        velocity = (signed - st.session_state.angles[-1]) / SAMPLE_PERIOD
    else:
        velocity = 0.0

    roll, pitch, yaw = quaternion_to_euler(q_motion)

    st.session_state.timestamps.append(elapsed)
    st.session_state.angles.append(signed)
    st.session_state.raw_fw_angles.append(sample["fw_ankle"])
    st.session_state.velocities.append(velocity)
    st.session_state.rel_roll.append(roll)
    st.session_state.rel_pitch.append(pitch)
    st.session_state.rel_yaw.append(yaw)
    st.session_state.foot_rpy.append(
        (sample["foot_roll"], sample["foot_pitch"], sample["foot_yaw"])
    )
    st.session_state.shank_rpy.append(
        (sample["shank_roll"], sample["shank_pitch"], sample["shank_yaw"])
    )

    st.session_state.current_velocity = velocity

    if st.session_state.min_angle is None:
        st.session_state.min_angle = signed
        st.session_state.max_angle = signed
    else:
        st.session_state.min_angle = min(st.session_state.min_angle, signed)
        st.session_state.max_angle = max(st.session_state.max_angle, signed)

    update_repetition(signed)


def ingest():
    """Pulls whatever is available and pushes it through the pipeline."""
    if st.session_state.simulation:
        lines = simulated_lines(int(LOOP_SLEEP * FIRMWARE_RATE_HZ))
    else:
        lines = read_serial_lines()

    data_count = 0

    for line in lines:
        sample = parse_data_line(line)

        if sample is None:
            # Status / banner text from the firmware
            if MARKER_NEUTRAL_SAVED in line.upper():
                st.session_state.awaiting_neutral = True
                log("Firmware: neutral position saved.")
            elif MARKER_GYRO_DONE in line.upper():
                log("Firmware: gyro calibration complete.")
            elif MARKER_READY.lower() in line.lower():
                st.session_state.device_ready = True
                log("Firmware: system ready.")
            elif line:
                log(f"FW> {line[:110]}")
            continue

        data_count += 1
        st.session_state.lines_received += 1

        if not st.session_state.device_ready:
            st.session_state.device_ready = True
            log("First data line parsed - device is streaming.")

        process_sample(sample)

    return data_count


# ============================================================
# SESSION CONTROL
# ============================================================

def calculate_score():
    """
    Accuracy = correct attempts / all attempts, as a percentage.
    Same formula as v2.0; shown to the patient as "Accuracy".
    """
    reps = st.session_state.repetitions

    if reps == 0:
        return 0.0

    return round(100.0 * st.session_state.correct_repetitions / reps, 1)


def achieved_rom():
    if st.session_state.min_angle is None:
        return 0.0

    return st.session_state.max_angle - st.session_state.min_angle


def start_session():
    reset_measurements()

    ser = st.session_state.serial_object
    if ser is not None:
        try:
            ser.reset_input_buffer()
            st.session_state.rx_buffer = ""
        except Exception:
            pass

    st.session_state.running = True
    log(f"Session started: {st.session_state.exercise}")


def stop_session():
    st.session_state.running = False
    st.session_state.session_complete = True
    log("Session stopped.")


def save_session():
    """
    Writes one row to `sessions` and one row per attempt to
    `repetitions`, in a single transaction. If anything fails, neither
    is saved, so there are never orphan repetition rows.
    """
    if not st.session_state.patient_id or not st.session_state.angles:
        return

    config = exercise_config()
    peaks = st.session_state.rep_peaks

    values = {
        "patient_id": st.session_state.patient_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exercise": st.session_state.exercise,
        "axis": config["axis"],
        "sign": config["sign"],
        "target_min": config["target_min"],
        "target_max": config["target_max"],
        "achieved_rom": achieved_rom(),
        "peak_angle": max(peaks) if peaks else float(max(st.session_state.angles)),
        "mean_peak": float(np.mean(peaks)) if peaks else 0.0,
        "repetitions": st.session_state.repetitions,
        "correct_repetitions": st.session_state.correct_repetitions,
        "incorrect_repetitions": st.session_state.incorrect_repetitions,
        "score": calculate_score(),
        "duration": st.session_state.sample_count * SAMPLE_PERIOD,
        "samples": st.session_state.sample_count,
        "source": "simulation" if st.session_state.simulation else "hardware",
        "notes": st.session_state.session_notes.strip(),
    }

    columns = ", ".join(values.keys())
    placeholders = ", ".join("?" for _ in values)

    conn = sqlite3.connect(DB_NAME)

    try:
        cursor = conn.cursor()
        cursor.execute(
            f"INSERT INTO sessions ({columns}) VALUES ({placeholders})",
            tuple(values.values()),
        )
        session_id = cursor.lastrowid
        st.session_state.last_saved_session_id = session_id

        rep_columns = ", ".join(name for name, _ in REPETITION_COLUMNS)
        rep_placeholders = ", ".join("?" for _ in REPETITION_COLUMNS)

        for rec in st.session_state.rep_records:
            cursor.execute(
                f"INSERT INTO repetitions ({rep_columns}) VALUES ({rep_placeholders})",
                (
                    session_id,
                    st.session_state.patient_id,
                    st.session_state.exercise,
                    rec["rep_number"],
                    rec["peak_angle"],
                    rec["duration_s"],
                    rec["result"],
                    rec["reason"],
                    rec["target_min"],
                    rec["target_max"],
                ),
            )

        conn.commit()
        log(f"Session saved (id {session_id}, "
            f"{len(st.session_state.rep_records)} repetitions).")

    except Exception as exc:
        conn.rollback()
        log(f"Save failed, nothing written: {exc}")
        st.error(f"Could not save the session: {exc}")

    finally:
        conn.close()


# The exact column order requested for the session CSV.
SESSION_CSV_COLUMNS = [
    "id", "patient_id", "date", "exercise", "axis", "sign",
    "target_min", "target_max", "achieved_rom", "peak_angle", "mean_peak",
    "repetitions", "correct_repetitions", "incorrect_repetitions",
    "score", "duration", "samples", "source", "notes",
]


def update_session_note(session_id, note):
    """
    Writes the note onto a session that was already auto-saved.

    The session saves itself the instant the last correct repetition
    lands, which is before the patient can type anything. So the note
    has to be attached afterwards, by row id.
    """
    if session_id is None:
        return False

    conn = sqlite3.connect(DB_NAME)

    try:
        conn.execute(
            "UPDATE sessions SET notes = ? WHERE id = ?",
            (note.strip(), session_id),
        )
        conn.commit()
        return True
    except Exception as exc:
        conn.rollback()
        log(f"Could not save note: {exc}")
        return False
    finally:
        conn.close()


def sessions_csv(patient_id=None, session_id=None):
    """
    ONE csv containing the session summary columns, in the requested
    order, plus notes.

    patient_id -> every session for that patient
    session_id -> just that one session
    """
    conn = sqlite3.connect(DB_NAME)

    query = "SELECT " + ", ".join(SESSION_CSV_COLUMNS) + " FROM sessions"
    clauses, params = [], []

    if session_id is not None:
        clauses.append("id = ?")
        params.append(session_id)
    elif patient_id is not None:
        clauses.append("patient_id = ?")
        params.append(patient_id)

    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    query += " ORDER BY date DESC, id DESC"

    frame = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Keep dates as readable ISO text in spreadsheet applications. The
    # leading tab forces Excel to treat the value as text rather than a
    # date-formatted numeric cell that may render as #### in a narrow column.
    if "date" in frame.columns:
        frame["date"] = frame["date"].fillna("").astype(str).map(
            lambda value: "\t" + value if value else ""
        )

    return frame.to_csv(index=False)


def session_dataframe():
    """Raw per-sample log, for offline analysis and later ML work."""
    if not st.session_state.angles:
        return pd.DataFrame()

    foot = np.array(st.session_state.foot_rpy)
    shank = np.array(st.session_state.shank_rpy)

    return pd.DataFrame({
        "time_s": st.session_state.timestamps,
        "movement_angle_deg": st.session_state.angles,
        "firmware_ankle_deg": st.session_state.raw_fw_angles,
        "angular_velocity_dps": st.session_state.velocities,
        "rel_roll_deg": st.session_state.rel_roll,
        "rel_pitch_deg": st.session_state.rel_pitch,
        "rel_yaw_deg": st.session_state.rel_yaw,
        "foot_roll": foot[:, 0],
        "foot_pitch": foot[:, 1],
        "foot_yaw": foot[:, 2],
        "shank_roll": shank[:, 0],
        "shank_pitch": shank[:, 1],
        "shank_yaw": shank[:, 2],
    })


def repetition_dataframe():
    """Per-attempt table for the current session (used for CSV + display)."""
    if not st.session_state.rep_records:
        return pd.DataFrame()

    return pd.DataFrame(st.session_state.rep_records)


def next_exercise_name():
    names = list(EXERCISES.keys())
    index = names.index(st.session_state.exercise)
    return names[(index + 1) % len(names)]


# ============================================================
# LOGIN PAGE
# ============================================================

def get_patient_record(patient_id):
    """Returns the stored profile for this patient ID, or None if it's new."""
    if not patient_id:
        return None

    conn = sqlite3.connect(DB_NAME)
    row = conn.execute(
        "SELECT patient_id, name, age, sex, created_at FROM patients WHERE patient_id = ?",
        (patient_id,),
    ).fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "patient_id": row[0],
        "name": row[1],
        "age": row[2],
        "sex": row[3],
        "created_at": row[4],
    }


def recent_sessions_preview(patient_id, limit=5):
    """A short session-history preview shown on the login page itself."""
    conn = sqlite3.connect(DB_NAME)
    frame = pd.read_sql_query(
        "SELECT date, exercise, correct_repetitions AS correct, "
        "incorrect_repetitions AS not_counted, score AS accuracy, "
        "achieved_rom AS range_deg "
        "FROM sessions WHERE patient_id = ? ORDER BY date DESC LIMIT ?",
        conn,
        params=(patient_id, limit),
    )
    conn.close()
    return frame


def log_in_as(patient_id, name, age, sex):
    st.session_state.patient_id = patient_id
    st.session_state.patient_name = name
    st.session_state.patient_age = age or 0
    st.session_state.patient_sex = sex or ""
    st.session_state.logged_in = True
    st.session_state.page = "Dashboard"


def login_page():
    st.title("Ankle Rehabilitation System")
    st.caption(
        "Wearable lower-limb physiotherapy guidance and monitoring - "
        "two-IMU ankle angle measurement"
    )

    patient_id = st.text_input(
        "Patient ID",
        key="login_patient_id",
        help="Always use the same ID to keep coming back to your own history.",
    ).strip()

    record = get_patient_record(patient_id)

    if record is not None:
        # Returning patient: no need to re-enter name/age/sex, and their
        # past sessions are shown right here before they continue.
        st.success(f"Welcome back, {record['name']}! 👋")

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Age**\n\n{record['age'] or '-'}")
        c2.markdown(f"**Sex**\n\n{record['sex'] or '-'}")
        c3.markdown(f"**Patient since**\n\n{(record['created_at'] or '-')[:10]}")

        st.markdown("##### Your recent sessions")
        preview = recent_sessions_preview(record["patient_id"])

        if preview.empty:
            st.caption("No sessions recorded yet - your first one will show up here.")
        else:
            st.dataframe(preview, use_container_width=True, hide_index=True)

        if st.button("Continue", type="primary", use_container_width=True):
            log_in_as(record["patient_id"], record["name"], record["age"], record["sex"])
            st.rerun()

        st.caption("Not you? Type a different Patient ID above.")

    elif patient_id:
        # New ID: collect the profile once, then it is remembered for
        # every future login with this same ID.
        st.info("This is a new Patient ID. Tell us a little about you to set up your profile.")

        col1, col2 = st.columns(2)

        with col1:
            patient_name = st.text_input("Patient name")

        with col2:
            age = st.number_input("Age", min_value=1, max_value=120, value=25)

        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)

        if st.button("Create profile & start", type="primary", use_container_width=True):
            if not patient_name:
                st.warning("Enter a patient name to continue.")
                return

            conn = sqlite3.connect(DB_NAME)
            conn.execute(
                "INSERT OR REPLACE INTO patients (patient_id, name, age, sex, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    patient_id,
                    patient_name,
                    age,
                    sex,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            conn.commit()
            conn.close()

            log_in_as(patient_id, patient_name, age, sex)
            st.rerun()
    else:
        st.info("Enter your Patient ID above to continue, or type a new one to create a profile.")


# ============================================================
# SIDEBAR
# ============================================================

PATIENT_NAV = [
    ("🏠  Home", "Dashboard"),
    ("▶️  Start session", "Live"),
    ("📈  My progress", "History"),
]

DEVELOPER_NAV = [
    ("Overview", "Dashboard"),
    ("Exercise", "Exercise"),
    ("Live session", "Live"),
    ("History", "History"),
    ("Settings", "Settings"),
]


def sidebar():
    # A previous Streamlit session may retain the old demo default after an
    # upgrade. Patient mode is always hardware-only.
    if not is_developer() and st.session_state.simulation:
        st.session_state.simulation = False
        disconnect_serial()
        reset_measurements()

    if is_developer():
        st.sidebar.markdown("### Developer workbench")
        st.sidebar.caption(
            f"patient {st.session_state.patient_id}  -  {st.session_state.patient_name}"
        )
    else:
        st.sidebar.markdown(f"### 👋 Hi, {st.session_state.patient_name}")
        sex_bit = f" · {st.session_state.patient_sex}" if st.session_state.patient_sex else ""
        st.sidebar.caption(f"Patient ID {st.session_state.patient_id}{sex_bit}")

    chosen = st.sidebar.radio(
        "View",
        ["Patient", "Developer"],
        index=0 if st.session_state.mode == "Patient" else 1,
        horizontal=False,
        help="Developer view shows serial settings, raw values and logs.",
    )

    if chosen != st.session_state.mode:
        if chosen == "Patient" and st.session_state.simulation:
            st.session_state.simulation = False
            disconnect_serial()
            reset_measurements()
        st.session_state.mode = chosen
        # Each mode has its own page set, so start it from its home page
        # rather than leaving the user on a page that may not exist there.
        st.session_state.page = "Dashboard"
        st.rerun()

    if is_developer():
        if st.session_state.simulation:
            st.sidebar.warning("Simulation mode")
        elif st.session_state.serial_connected:
            st.sidebar.success(f"Connected: {st.session_state.serial_port}")
        else:
            st.sidebar.error("No device connected")

    st.sidebar.divider()

    pages = DEVELOPER_NAV if is_developer() else PATIENT_NAV

    for label, page in pages:
        active = st.session_state.page == page
        if st.sidebar.button(
            label,
            key=f"nav_{page}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state.page = page
            st.rerun()

    st.sidebar.divider()

    if st.sidebar.button("Log out", key="nav_logout", use_container_width=True):
        disconnect_serial()
        st.session_state.logged_in = False
        st.session_state.page = "Login"
        st.rerun()


# ============================================================
# DASHBOARD
#
# CHANGED: totals now come from SQLite, not session state. In v2.0 a
# fresh login always showed 0 repetitions and 0% score.
# ============================================================

def patient_totals():
    conn = sqlite3.connect(DB_NAME)

    row = conn.execute(
        """
        SELECT COUNT(*),
               COALESCE(SUM(correct_repetitions), 0),
               COALESCE(SUM(repetitions), 0),
               COALESCE(MAX(achieved_rom), 0)
        FROM sessions WHERE patient_id = ?
        """,
        (st.session_state.patient_id,),
    ).fetchone()

    conn.close()

    sessions, correct, attempted, best_rom = row
    accuracy = round(100.0 * correct / attempted, 1) if attempted else 0.0

    return {
        "sessions": sessions,
        "correct": correct,
        "attempted": attempted,
        "accuracy": accuracy,
        "best_rom": best_rom,
    }


# ---- engagement data: everything below is computed from real saved sessions ----

# Which exercises are "completed" = a saved session with all target reps correct.
EXERCISE_STYLE = {
    "Dorsiflexion":   {"css": "ex-dorsi", "emoji": "⬆️", "blurb": "Lift the foot toward your shin"},
    "Plantarflexion": {"css": "ex-plant", "emoji": "⬇️", "blurb": "Point the foot away from you"},
    "Inversion":      {"css": "ex-inver", "emoji": "↪️", "blurb": "Roll the sole inward"},
    "Eversion":       {"css": "ex-ever",  "emoji": "↩️", "blurb": "Roll the sole outward"},
}


def completed_exercises_today():
    """
    Names of exercises this patient fully completed today (real DB rows).
    "Completed" means a saved session whose correct repetitions reached
    that exercise's own target, so changing a target in EXERCISES is
    picked up automatically.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute(
        """
        SELECT exercise, MAX(correct_repetitions) FROM sessions
        WHERE patient_id = ? AND date LIKE ?
        GROUP BY exercise
        """,
        (st.session_state.patient_id, today + "%"),
    ).fetchall()
    conn.close()

    return {
        name for name, best in rows
        if name in EXERCISES and (best or 0) >= EXERCISES[name]["target_reps"]
    }


def practice_streak():
    """
    Consecutive days, ending today or yesterday, on which the patient
    saved at least one session. Missing today does not break the streak
    until the day is over.
    """
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute(
        "SELECT DISTINCT substr(date, 1, 10) FROM sessions WHERE patient_id = ?",
        (st.session_state.patient_id,),
    ).fetchall()
    conn.close()

    days = set()
    for (text,) in rows:
        try:
            days.add(datetime.strptime(text, "%Y-%m-%d").date())
        except (TypeError, ValueError):
            continue

    if not days:
        return 0

    from datetime import timedelta
    cursor = datetime.now().date()

    if cursor not in days:
        cursor -= timedelta(days=1)

    streak = 0
    while cursor in days:
        streak += 1
        cursor -= timedelta(days=1)

    return streak


def earned_badges(totals, streak, done_today):
    """
    Badge list as (icon, name, description, earned). Each rule is
    a plain check on saved data, so a badge is only ever shown as earned
    when the data really supports it.
    """
    return [
        ("🌱", "First step", "Finish your first session", totals["sessions"] >= 1),
        ("🔥", "3-day streak", "Practise 3 days in a row", streak >= 3),
        ("🏆", "Full set", "Finish all 4 exercises in one day", len(done_today) >= 4),
        ("🎯", "Sharp shooter", "Reach 90% accuracy over 50+ attempts",
         totals["attempted"] >= 50 and totals["accuracy"] >= 90),
        ("💪", "100 club", "Log 100 correct repetitions", totals["correct"] >= 100),
    ]


def daily_goal_percent(done_today):
    """Share of the four exercises completed today."""
    return int(round(100 * len(done_today) / len(EXERCISES)))


def dashboard_page():
    if is_developer():
        developer_dashboard()
        return

    totals = patient_totals()
    streak = practice_streak()
    done_today = completed_exercises_today()
    goal = daily_goal_percent(done_today)
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"

    if goal >= 100:
        headline = "All four exercises done today. Brilliant!"
    elif goal > 0:
        headline = f"{len(done_today)} of {len(EXERCISES)} exercises done today. Keep going!"
    else:
        headline = "Ready for today's exercises? Let's get moving."

    st.markdown(
        f"""
        <div class="p-hero">
            <div class="ring" style="--p:{goal}"><span>{goal}%<small>today</small></span></div>
            <div class="txt">
                <h2>{greeting}, {st.session_state.patient_name}!</h2>
                <p>{headline}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="tiles">
            <div class="tile gold"><div class="v">🔥 {streak}</div>
                <div class="l">day streak</div></div>
            <div class="tile teal"><div class="v">{totals['sessions']}</div>
                <div class="l">{"session" if totals['sessions'] == 1 else "sessions"} completed</div></div>
            <div class="tile coral"><div class="v">{totals['correct']}</div>
                <div class="l">correct repetitions</div></div>
            <div class="tile blue"><div class="v">{totals['accuracy']:.0f}%</div>
                <div class="l">overall accuracy</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Today's exercises")

    columns = st.columns(len(EXERCISES))
    for column, (name, style) in zip(columns, EXERCISE_STYLE.items()):
        cfg = EXERCISES[name]
        finished = name in done_today
        tag = "✅ Done today" if finished else f"{cfg['target_reps']} correct reps"

        with column:
            st.markdown(
                f"""
                <div class="ex-card {style['css']}">
                    <h4>{name}</h4>
                    <p>{style['blurb']}</p>
                    <span class="tag">{tag}</span>
                    <span class="emoji">{style['emoji']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "Review" if finished else "Start",
                key=f"dash_start_{name}",
                use_container_width=True,
                type="secondary" if finished else "primary",
            ):
                # Just selects the exercise; details/video/live button for it
                # are shown further down this same Home page.
                st.session_state.exercise = name
                st.session_state.exercise_quote = random.choice(MOTIVATION_QUOTES)
                reset_measurements()
                st.rerun()

    st.subheader("Your badges")

    cards = ""
    for icon, name, desc, earned in earned_badges(totals, streak, done_today):
        locked = "" if earned else " locked"
        cards += (
            f'<div class="badge{locked}"><div class="ic">{icon}</div>'
            f'<div class="nm">{name}</div><div class="ds">{desc}</div></div>'
        )
    st.markdown(f'<div class="badges">{cards}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="quote-card">{random.choice(MOTIVATION_QUOTES)}</div>',
        unsafe_allow_html=True,
    )

    # ---- merged-in exercise details (was a separate "Exercise" page) ----
    # Shows the currently selected exercise (from the cards above) with its
    # target, instructions and tutorial video, ending in the button that
    # goes to the live session - all on this same Home page.
    st.divider()

    selected = st.session_state.exercise
    config = exercise_config(selected)
    style = EXERCISE_STYLE.get(selected, EXERCISE_STYLE["Dorsiflexion"])

    st.subheader(f"{style['emoji']} {selected}")

    st.markdown(
        f"""
        <div class="ex-card {style['css']} chosen">
            <h4>{selected}</h4>
            <p>{config['description']}</p>
            <span class="tag">Aim for {config['target_min']}-{config['target_max']} deg</span>
            <span class="tag">{config['target_reps']} correct reps</span>
            <span class="emoji">{style['emoji']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.exercise_quote:
        st.session_state.exercise_quote = random.choice(MOTIVATION_QUOTES)

    st.markdown(
        f'<div class="quote-card">{st.session_state.exercise_quote}</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])

    with left:
        st.markdown("##### How to do it")
        st.markdown(
            "1. Sit with the shank supported and the foot free.\n"
            "2. Strap the foot sensor to the top of the foot and the shank "
            "sensor to the front of the lower leg, both aligned with the "
            "limb.\n"
            "3. Hold the ankle in a relaxed neutral position and capture it.\n"
            "4. Move slowly to the prescribed range.\n"
            "5. Return fully to neutral - a repetition is only counted on "
            "the return.\n"
            "6. Repeat until you have completed the required number of "
            "correct repetitions."
        )

    with right:
        st.markdown("##### Tutorial video")
        show_tutorial_video(selected)

    if st.button("▶️  Go to live session", type="primary", use_container_width=True):
        st.session_state.page = "Live"
        st.rerun()


def developer_dashboard():
    """Compact technical overview. Deliberately unlike the patient home."""
    config = exercise_config()
    totals = patient_totals()

    st.markdown(
        f'<div class="dev-banner"><b>DEVELOPER VIEW</b> &nbsp;|&nbsp; '
        f'patient {st.session_state.patient_id} &nbsp;|&nbsp; '
        f'exercise {st.session_state.exercise} &nbsp;|&nbsp; '
        f'axis {config["axis"]} &nbsp;|&nbsp; sign {config["sign"]:+d} &nbsp;|&nbsp; '
        f'target {config["target_min"]}-{config["target_max"]} deg</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Sessions", totals["sessions"])
    col2.metric("Correct reps", totals["correct"])
    col3.metric("Attempts", totals["attempted"])
    col4.metric("Accuracy", f"{totals['accuracy']}%")
    col5.metric("Best ROM", f"{totals['best_rom']:.1f} deg")

    conn = sqlite3.connect(DB_NAME)
    history = pd.read_sql_query(
        "SELECT id, date, exercise, source, repetitions, correct_repetitions, "
        "incorrect_repetitions, score, achieved_rom, samples "
        "FROM sessions WHERE patient_id = ? ORDER BY date DESC LIMIT 10",
        conn,
        params=(st.session_state.patient_id,),
    )
    conn.close()

    st.subheader("Last 10 sessions")

    if history.empty:
        st.info("No sessions recorded for this patient yet.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)


# ============================================================
# EXERCISE PAGE
#
# CHANGED: tutorial video slot and a motivational quote.
# ============================================================

def show_tutorial_video(name):
    url = VIDEO_URLS.get(name, "")

    if url:
        st.video(url)
        st.caption(
            "Tutorial video. Watch it once before you begin, and follow "
            "your physiotherapist's advice if it differs."
        )
    else:
        st.info(
            "The tutorial video for this exercise will appear here once "
            "it has been added."
        )


def exercise_page():
    if is_developer():
        st.title("Exercise selection")
    else:
        st.markdown("## 🦶 Choose your exercise")

    names = list(EXERCISES.keys())

    selected = st.selectbox(
        "Exercise",
        names,
        index=names.index(st.session_state.exercise),
    )

    if selected != st.session_state.exercise:
        st.session_state.exercise = selected
        st.session_state.exercise_quote = random.choice(MOTIVATION_QUOTES)

    if not st.session_state.exercise_quote:
        st.session_state.exercise_quote = random.choice(MOTIVATION_QUOTES)

    config = exercise_config(selected)

    if is_developer():
        st.write(config["description"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Minimum", f"{config['target_min']} deg")
        col2.metric("Maximum", f"{config['target_max']} deg")
        col3.metric("Correct reps needed", config["target_reps"])
    else:
        style = EXERCISE_STYLE.get(selected, EXERCISE_STYLE["Dorsiflexion"])
        st.markdown(
            f"""
            <div class="ex-card {style['css']} chosen">
                <h4>{selected}</h4>
                <p>{config['description']}</p>
                <span class="tag">Aim for {config['target_min']}-{config['target_max']} deg</span>
                <span class="tag">{config['target_reps']} correct reps</span>
                <span class="emoji">{style['emoji']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not is_developer():
        st.markdown(
            f'<div class="quote-card">{st.session_state.exercise_quote}</div>',
            unsafe_allow_html=True,
        )

    if is_developer():
        st.caption(f"Measurement axis: {config['axis']}  |  sign: {config['sign']:+d}")

        if config["axis"] != "firmware":
            st.info(
                "This exercise is measured from the reconstructed relative "
                "orientation rather than the firmware's Ankle value. Confirm "
                "the axis choice in Settings before trusting the numbers."
            )

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("How to do it")
        st.markdown(
            "1. Sit with the shank supported and the foot free.\n"
            "2. Strap the foot sensor to the top of the foot and the shank "
            "sensor to the front of the lower leg, both aligned with the "
            "limb.\n"
            "3. Hold the ankle in a relaxed neutral position and capture it.\n"
            "4. Move slowly to the prescribed range.\n"
            "5. Return fully to neutral - a repetition is only counted on "
            "the return.\n"
            "6. Repeat until you have completed the required number of "
            "correct repetitions."
        )

    with right:
        st.subheader("Tutorial video")
        show_tutorial_video(selected)

    if st.button("Go to live session", type="primary"):
        st.session_state.page = "Live"
        st.rerun()


# ============================================================
# LIVE SESSION PAGE
# ============================================================

def connection_panel():
    """Developer-only: serial and simulation controls."""
    st.subheader("Device")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        simulation = st.checkbox(
            "Simulation mode",
            value=st.session_state.simulation,
            help="Generates firmware-format lines so the full pipeline "
                 "can be tested without hardware.",
        )

        if simulation != st.session_state.simulation:
            st.session_state.simulation = simulation
            if simulation:
                disconnect_serial()
                st.session_state.calibrated = True
            st.rerun()

    if st.session_state.simulation:
        with col2:
            st.info("Running on synthetic data.")

        with col3:
            options = ["Correct", "Too short", "Too far", "Mixed"]
            current = st.session_state.get("sim_quality", "Correct")
            st.selectbox(
                "Simulated movement quality",
                options,
                index=options.index(current),
                key="sim_quality",
                help="Choose how the simulated patient moves, to test how "
                     "correct and incorrect repetitions are recorded.",
            )
        return

    ports = available_ports()

    with col2:
        if ports:
            port = st.selectbox("Serial port", ports)
        else:
            port = None
            st.warning(
                "No serial ports found. Check the USB cable and the "
                "CP2102 / CH340 driver."
            )

    with col3:
        baudrate = st.selectbox(
            "Baud rate",
            [115200, 230400, 9600],
            index=0,
            help="The firmware uses 115200.",
        )

    button_col1, button_col2 = st.columns(2)

    with button_col1:
        if st.button("Connect", use_container_width=True, disabled=port is None):
            if connect_serial(port, baudrate):
                st.rerun()

    with button_col2:
        if st.button(
            "Disconnect",
            use_container_width=True,
            disabled=not st.session_state.serial_connected,
        ):
            disconnect_serial()
            st.rerun()

    if not st.session_state.serial_connected:
        return

    if not st.session_state.device_ready:
        elapsed = time.time() - (st.session_state.connect_time or time.time())
        remaining = max(0, DEVICE_BOOT_SECONDS - elapsed)

        if remaining > 0:
            st.warning(
                f"Waiting for the ESP32 to finish booting and calibrating "
                f"its gyroscopes. Keep both IMUs completely still. "
                f"About {remaining:.0f} s remaining."
            )
        else:
            st.error(
                f"Connected for {elapsed:.0f} s but no readable data line "
                f"has arrived yet. Open the serial log below to see the raw "
                f"text. Check the baud rate, and make sure the Arduino "
                f"Serial Monitor is closed."
            )

        if st.button("Continue anyway"):
            st.session_state.device_ready = True
            log("Device marked ready manually.")
            st.rerun()
    else:
        st.success(
            f"Device streaming. {st.session_state.lines_received} samples "
            f"received."
        )


def patient_connection_status():
    """Patient-facing device status: one line, no jargon."""
    if not st.session_state.serial_connected:
        st.warning("No sensor connected. Connect the ESP32 to begin a real session.")
    elif not st.session_state.device_ready:
        st.warning("Sensors are starting up. Please keep your foot still.")
    else:
        st.success("Sensors connected and ready.")


def patient_connection_panel():
    """Simple hardware connection controls available in Patient mode."""
    st.subheader("Sensor connection")
    if not SERIAL_AVAILABLE:
        st.error("Serial support is missing. Install it with: python -m pip install pyserial")
        return

    if st.session_state.serial_connected:
        st.success(f"Connected to {st.session_state.serial_port}")
        if st.session_state.device_ready:
            st.caption("Sensor data is being received.")
        else:
            st.caption("Waiting for the ESP32 to finish starting. Keep both IMUs still.")
        if st.button("Disconnect sensors", key="patient_disconnect", use_container_width=True):
            disconnect_serial()
            reset_measurements()
            st.rerun()
        return

    ports = available_ports()
    if not ports:
        st.warning("No serial port found. Connect the ESP32 using USB and check its driver.")
        return

    col_port, col_connect = st.columns([3, 1])
    with col_port:
        port = st.selectbox("ESP32 serial port", ports, key="patient_serial_port")
    with col_connect:
        st.write("")
        st.write("")
        if st.button("Connect", key="patient_connect", type="primary", use_container_width=True):
            if connect_serial(port, DEFAULT_BAUDRATE):
                st.rerun()


def calibration_panel():
    st.subheader("Neutral position")

    if not st.session_state.simulation and not st.session_state.serial_connected:
        st.info("Connect the device first.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button(
            "Capture neutral",
            use_container_width=True,
            disabled=not st.session_state.simulation
            and not st.session_state.device_ready,
        ):
            if st.session_state.simulation:
                st.session_state.awaiting_neutral = True
                log("Simulation: neutral capture requested.")
            else:
                send_neutral_command()
            st.rerun()

    with col2:
        if st.session_state.awaiting_neutral:
            st.info(
                "Hold the ankle still. The firmware pauses for about two "
                "seconds while it stores the neutral orientation."
            )
        elif st.session_state.calibrated:
            st.success("Neutral captured. The current position reads zero.")
        else:
            st.warning(
                "Rest your foot in a relaxed neutral position, then press "
                "Capture neutral. Angles are meaningless until this is done."
            )


def patient_calibration_panel():
    """Patient-facing neutral capture; sends the firmware's N command."""
    if not st.session_state.serial_connected:
        st.info("Connect the ESP32 before setting your neutral position.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button(
            "📍 Set neutral position",
            use_container_width=True,
            disabled=not st.session_state.device_ready,
            key="patient_capture_neutral",
        ):
            send_neutral_command()
            st.rerun()

    with col2:
        if st.session_state.awaiting_neutral:
            st.info("Hold still for a moment while we set your starting position...")
        elif st.session_state.calibrated:
            st.success("Your neutral position is set.")
        else:
            st.markdown(
                "Rest your foot in a relaxed, natural position, then press the "
                "button. This becomes your starting point."
            )


def polarity_panel():
    """
    Developer-only. Lets the student confirm the sign convention on
    their own hardware before trusting the repetition counter.
    """
    with st.expander("Polarity check"):
        st.write(
            "Move the foot into the selected exercise and read the raw "
            "value below. If it goes negative, flip the sign for that "
            "exercise in Settings."
        )

        sample = st.session_state.current_sample

        if sample is None:
            st.caption("No samples received yet.")
            return

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Firmware ankle", f"{sample['fw_ankle']:.2f}")

        if st.session_state.rel_roll:
            col2.metric("Relative roll", f"{st.session_state.rel_roll[-1]:.2f}")
            col3.metric("Relative pitch", f"{st.session_state.rel_pitch[-1]:.2f}")
            col4.metric("Relative yaw", f"{st.session_state.rel_yaw[-1]:.2f}")

        st.caption(
            "Relative roll should track the firmware ankle value closely. "
            "A large disagreement means the neutral captured on the GUI "
            "side and the one stored in firmware were taken at different "
            "moments - recapture neutral."
        )


def live_chart():
    """Real-time angle trace with the target band shaded, plus a
    velocity trace underneath so the therapist can see how briskly the
    patient is moving, not just how far."""
    if len(st.session_state.timestamps) < 2:
        return

    config = exercise_config()

    window = int(PLOT_WINDOW_SECONDS * FIRMWARE_RATE_HZ)
    times = st.session_state.timestamps[-window:]
    angles = st.session_state.angles[-window:]
    velocities = st.session_state.velocities[-window:]

    step = max(1, len(times) // PLOT_MAX_POINTS)
    times = times[::step]
    angles = angles[::step]
    velocities = velocities[::step]

    # Patients see the angle trace only; the velocity trace is a
    # developer / clinician tool.
    if is_developer():
        figure, (ax_angle, ax_vel) = plt.subplots(
            2, 1, figsize=(9, 4.6), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1]},
        )
    else:
        figure, ax_angle = plt.subplots(figsize=(9, 3.2))
        ax_vel = None

    ax_angle.axhspan(
        config["target_min"], config["target_max"],
        color="#2ecc71", alpha=0.15, label="Target band",
    )
    ax_angle.plot(times, angles, color="#1f77b4", linewidth=1.4, label="Angle")
    ax_angle.axhline(config["target_min"], color="#2ecc71", linewidth=0.8, linestyle="--")
    ax_angle.axhline(config["target_max"], color="#2ecc71", linewidth=0.8, linestyle="--")
    ax_angle.axhline(0, color="#999999", linewidth=0.6)
    ax_angle.set_ylabel("Angle (deg)")
    ax_angle.legend(loc="upper right", fontsize=8)
    ax_angle.grid(True, alpha=0.25)

    if ax_vel is not None:
        ax_vel.plot(times, velocities, color="#e67e22", linewidth=1.0)
        ax_vel.axhline(0, color="#999999", linewidth=0.6)
        ax_vel.set_ylabel("Velocity (deg/s)")
        ax_vel.set_xlabel("Time (s)")
        ax_vel.grid(True, alpha=0.25)
    else:
        ax_angle.set_xlabel("Time (s)")

    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


def build_rep_history_png(records, target_min, target_max):
    """
    Draws the attempt bar chart and returns it as PNG bytes.

    Split out from rep_history_chart() so the result can be cached:
    the chart only changes when a new attempt is recorded, but the
    Live page reruns about 20 times a second. Drawing a matplotlib
    figure on every rerun was the slowest part of the page.
    """
    peaks = [r["peak_angle"] for r in records]
    colors = ["#2ecc71" if r["result"] == "correct" else "#e74c3c" for r in records]
    rep_numbers = [r["rep_number"] for r in records]

    figure, axis = plt.subplots(figsize=(9, 2.6))
    axis.bar(rep_numbers, peaks, color=colors)
    axis.axhspan(target_min, target_max, color="#2ecc71", alpha=0.12)
    axis.axhline(target_min, color="#2ecc71", linewidth=0.8, linestyle="--")
    axis.axhline(target_max, color="#2ecc71", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Attempt")
    axis.set_ylabel("Peak angle (deg)")
    axis.set_xticks(rep_numbers)
    axis.grid(True, alpha=0.25, axis="y")
    figure.tight_layout()

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100)
    plt.close(figure)

    return buffer.getvalue()


def rep_history_chart():
    """Bar chart of each recorded attempt's peak angle, coloured by
    correct/incorrect, with the target band overlaid - an at-a-glance
    view of consistency across the set.

    The PNG is rebuilt only when the number of attempts or the target
    range changes; otherwise the stored image is reused."""
    records = st.session_state.rep_records

    if not records:
        return

    config = exercise_config()
    key = (len(records), config["target_min"], config["target_max"])

    if st.session_state.get("rep_chart_key") != key:
        st.session_state.rep_chart_png = build_rep_history_png(
            records, config["target_min"], config["target_max"]
        )
        st.session_state.rep_chart_key = key

    st.image(st.session_state.rep_chart_png, width="stretch")


def patient_instruction(config):
    """
    One large line telling the patient what to do right now. Uses the
    same state the developer sees, but says it in plain words.
    """
    angle = st.session_state.current_angle
    state = st.session_state.movement_state

    if state == "MOVING":
        if angle < config["target_min"]:
            return "instruction-more", "Keep going - a little further! 💪"
        if angle > config["target_max"]:
            return "instruction-back", "That's far enough - ease back. 🙌"
        return "instruction-good", "Perfect! Hold it, then return. ✨"

    if st.session_state.last_rep_correct is False:
        return "instruction-more", "Return to neutral, then try again."

    if st.session_state.last_rep_correct is True:
        return "instruction-neutral", "Nice one! Relax, then go again."

    return "instruction-neutral", "Relax at neutral, then begin your first repetition."


def rep_dots_html(config):
    """
    One dot per required correct repetition. Filled dots are earned reps,
    the ringed dot is the next one to earn. Counts CORRECT reps only,
    the same number that decides completion.
    """
    total = config["target_reps"]
    done = min(st.session_state.correct_repetitions, total)

    dots = ""
    for i in range(1, total + 1):
        if i <= done:
            dots += f'<div class="dot done">{i}</div>'
        elif i == done + 1:
            dots += f'<div class="dot next">{i}</div>'
        else:
            dots += f'<div class="dot">{i}</div>'

    return (
        f'<div class="dots">{dots}</div>'
        f'<div class="dots-cap">{done} of {total} correct repetitions</div>'
    )


def angle_gauge_html(config):
    """
    Big live angle number over a horizontal track, with the target range
    shaded green. The bar fills as the foot moves, so the patient can see
    where they are relative to the goal without reading a graph.
    """
    angle = st.session_state.current_angle
    target_min = config["target_min"]
    target_max = config["target_max"]

    # Scale runs from 0 to a little past the top of the target range.
    scale_max = max(target_max * 1.3, target_max + 5.0)

    def pct(value):
        return max(0.0, min(100.0, 100.0 * value / scale_max))

    fill = pct(angle)
    band_left = pct(target_min)
    band_width = pct(target_max) - band_left

    return (
        f'<div class="gauge-wrap">'
        f'<div class="gauge-num">{max(angle, 0):.0f}<small> deg</small></div>'
        f'<div class="gauge-track">'
        f'<div class="gauge-target" style="left:{band_left:.1f}%;width:{band_width:.1f}%"></div>'
        f'<div class="gauge-fill" style="width:{fill:.1f}%"></div>'
        f'</div>'
        f'<div class="gauge-scale"><span>0</span>'
        f'<span>Aim for {target_min}-{target_max} deg</span>'
        f'<span>{scale_max:.0f}</span></div>'
        f'</div>'
    )


def _confetti_burst():
    """Falling, tumbling paper confetti in mixed colours."""
    components.html(
        """
        <div id="confetti-root" style="position:fixed;inset:0;pointer-events:none;z-index:9999;overflow:hidden;"></div>
        <script>
        (function () {
            const root = document.getElementById('confetti-root');
            const colors = ['#f43f5e', '#f59e0b', '#22c55e', '#0ea5e9', '#a855f7', '#fb7185'];
            for (let i = 0; i < 130; i++) {
                const piece = document.createElement('div');
                const size = 6 + Math.random() * 6;
                piece.style.position = 'absolute';
                piece.style.width = size + 'px';
                piece.style.height = (size * 0.4) + 'px';
                piece.style.background = colors[Math.floor(Math.random() * colors.length)];
                piece.style.left = (Math.random() * 100) + 'vw';
                piece.style.top = '-20px';
                piece.style.opacity = '0.95';
                piece.style.borderRadius = '2px';
                piece.style.transform = 'rotate(' + (Math.random() * 360) + 'deg)';
                const duration = 2.2 + Math.random() * 1.6;
                piece.style.transition =
                    'transform ' + duration + 's ease-in, top ' + duration + 's ease-in, opacity ' + duration + 's ease-in';
                root.appendChild(piece);
                setTimeout(() => {
                    piece.style.top = (100 + Math.random() * 10) + 'vh';
                    piece.style.transform = 'rotate(' + (720 + Math.random() * 360) + 'deg)';
                    piece.style.opacity = '0';
                }, 30);
                setTimeout(() => piece.remove(), duration * 1000 + 300);
            }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def _fireworks_burst():
    """A few expanding radial bursts, like fireworks seen from below."""
    components.html(
        """
        <div id="fw-root" style="position:fixed;inset:0;pointer-events:none;z-index:9999;overflow:hidden;"></div>
        <script>
        (function () {
            const root = document.getElementById('fw-root');
            const colors = ['#fbbf24', '#fb7185', '#38bdf8', '#a3e635', '#c084fc'];

            function burst(cx, cy) {
                const color = colors[Math.floor(Math.random() * colors.length)];
                for (let i = 0; i < 26; i++) {
                    const angle = (i / 26) * 2 * Math.PI;
                    const dist = 60 + Math.random() * 70;
                    const spark = document.createElement('div');
                    spark.style.position = 'absolute';
                    spark.style.left = cx + 'px';
                    spark.style.top = cy + 'px';
                    spark.style.width = '6px';
                    spark.style.height = '6px';
                    spark.style.borderRadius = '50%';
                    spark.style.background = color;
                    spark.style.boxShadow = '0 0 6px ' + color;
                    spark.style.transition = 'transform 0.9s ease-out, opacity 0.9s ease-out';
                    root.appendChild(spark);
                    requestAnimationFrame(() => {
                        spark.style.transform =
                            'translate(' + (Math.cos(angle) * dist) + 'px,' + (Math.sin(angle) * dist) + 'px)';
                        spark.style.opacity = '0';
                    });
                    setTimeout(() => spark.remove(), 1000);
                }
            }

            const w = window.innerWidth, h = window.innerHeight;
            burst(w * 0.28, h * 0.35);
            setTimeout(() => burst(w * 0.72, h * 0.3), 280);
            setTimeout(() => burst(w * 0.5, h * 0.55), 560);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def _sparkle_burst():
    """Twinkling stars scattered over the screen."""
    components.html(
        """
        <div id="sp-root" style="position:fixed;inset:0;pointer-events:none;z-index:9999;overflow:hidden;"></div>
        <script>
        (function () {
            const root = document.getElementById('sp-root');
            const stars = ['\\u2728', '\\u2b50', '\\ud83c\\udf1f', '\\ud83d\\udcab'];
            for (let i = 0; i < 36; i++) {
                const s = document.createElement('div');
                s.textContent = stars[Math.floor(Math.random() * stars.length)];
                s.style.position = 'absolute';
                s.style.left = (Math.random() * 100) + 'vw';
                s.style.top = (Math.random() * 100) + 'vh';
                s.style.fontSize = (14 + Math.random() * 18) + 'px';
                s.style.opacity = '0';
                s.style.transition = 'opacity 0.6s ease-in-out, transform 1.2s ease-in-out';
                root.appendChild(s);
                setTimeout(() => {
                    s.style.opacity = '1';
                    s.style.transform = 'scale(1.3) rotate(20deg)';
                }, 30 + i * 20);
                setTimeout(() => { s.style.opacity = '0'; }, 1200 + i * 20);
                setTimeout(() => s.remove(), 2000 + i * 20);
            }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def celebrate():
    """
    Fires one of five motivational effects, chosen at random. Called
    exactly once per completed session (see the celebration_shown flag).
    """
    effect = random.choice(CELEBRATION_EFFECTS)

    if effect == "balloons":
        st.balloons()
    elif effect == "snow":
        st.snow()
    elif effect == "confetti":
        _confetti_burst()
    elif effect == "fireworks":
        _fireworks_burst()
    else:
        _sparkle_burst()


def completion_panel(config):
    """
    Shown when the target number of correct repetitions is reached.
    Completion is unconditional on mistakes: mistakes only lower the
    accuracy figure, they never block completion.
    """
    if not st.session_state.completion_quote:
        st.session_state.completion_quote = random.choice(MOTIVATION_QUOTES)

    # One of five celebration effects fires exactly once per completed session.
    if not st.session_state.celebration_shown:
        celebrate()
        st.session_state.celebration_shown = True

    st.markdown(
        f"""
        <div class="complete-card">
            <h2>Exercise Completed! 🎉</h2>
            <p>You have successfully completed all {config['target_reps']} repetitions.</p>
            <p>Your exercise procedure and repetitions were performed correctly.</p>
            <p><b>Super! You're performing very well. Keep up the great work! 💪</b></p>
            <span class="stat-pill">Accuracy {calculate_score():.0f}%</span>
            <span class="stat-pill">{st.session_state.correct_repetitions} correct</span>
            <span class="stat-pill">{st.session_state.incorrect_repetitions} not counted</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="quote-card">{st.session_state.completion_quote}</div>',
        unsafe_allow_html=True,
    )

    next_name = next_exercise_name()

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            f"Start next exercise: {next_name}",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.exercise = next_name
            st.session_state.exercise_quote = random.choice(MOTIVATION_QUOTES)
            reset_measurements()
            st.session_state.page = "Exercise" if is_developer() else "Dashboard"
            st.rerun()

    with col2:
        if st.button("Finish session", use_container_width=True):
            reset_measurements()
            st.session_state.page = "History"
            st.rerun()


def notes_box(key_prefix):
    """
    Free-text note for the finished session. The session has already
    been saved automatically, so pressing the button attaches the note
    to that saved row.
    """
    session_id = st.session_state.last_saved_session_id

    if session_id is None:
        return

    st.subheader("Add a note")
    st.caption(
        "For example: how the ankle felt, any pain, or something to tell "
        "your physiotherapist."
    )

    note = st.text_area(
        "Session note",
        value=st.session_state.session_notes,
        key=f"{key_prefix}_note_input",
        height=90,
        label_visibility="collapsed",
        placeholder="Optional note about this session...",
    )

    if st.button("Save note", key=f"{key_prefix}_note_save"):
        st.session_state.session_notes = note

        if update_session_note(session_id, note):
            st.success("Note saved with this session.")
        else:
            st.error("The note could not be saved. Try again.")


def session_downloads(key_prefix):
    """
    The single combined CSV (all session columns + notes), available in
    both modes. Row-level and raw-sample files are developer-only.
    """
    session_id = st.session_state.last_saved_session_id

    if session_id is not None:
        st.download_button(
            "Download this session (CSV)",
            sessions_csv(session_id=session_id),
            file_name=(
                f"{st.session_state.patient_id}_"
                f"{st.session_state.exercise}_session{session_id}.csv"
            ),
            mime="text/csv",
            key=f"{key_prefix}_dl_session",
        )

    if not is_developer():
        return

    frame = session_dataframe()

    if not frame.empty:
        st.download_button(
            "Download raw sample log (CSV)",
            frame.to_csv(index=False),
            file_name=(
                f"{st.session_state.patient_id}_"
                f"{st.session_state.exercise}_samples_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            ),
            mime="text/csv",
            key=f"{key_prefix}_dl_samples",
        )

    reps = repetition_dataframe()

    if not reps.empty:
        st.download_button(
            "Download repetition table (CSV)",
            reps.to_csv(index=False),
            file_name=(
                f"{st.session_state.patient_id}_"
                f"{st.session_state.exercise}_reps_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            ),
            mime="text/csv",
            key=f"{key_prefix}_dl_reps",
        )


def session_summary(config):
    st.subheader("Session summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Attempts", st.session_state.repetitions)
    col2.metric("Correct", st.session_state.correct_repetitions)
    col3.metric("Not counted", st.session_state.incorrect_repetitions)
    col4.metric("Accuracy", f"{calculate_score()}%")

    reps = repetition_dataframe()

    if not reps.empty:
        st.write("Every attempt in this session")

        shown = reps.copy()
        shown["result"] = shown["result"].map(
            {"correct": "Correct", "incorrect": "Not counted"}
        )
        shown["reason"] = shown["reason"].map(
            {"ok": "Good range", "too_short": "Too short", "too_far": "Too far"}
        )
        shown = shown[["rep_number", "peak_angle", "duration_s", "result", "reason"]]
        shown.columns = ["Attempt", "Peak (deg)", "Time (s)", "Result", "Note"]

        # Sized to fit every row, so no attempts are hidden behind an
        # inner scrollbar (about 38 px per row, capped for very long sets).
        st.dataframe(
            shown,
            use_container_width=True,
            hide_index=True,
            height=min(38 * (len(shown) + 1), 900),
        )

    notes_box("summary")
    session_downloads("summary")


def run_acquisition():
    """
    Shared by both views, so the patient and developer screens can never
    behave differently. Pulls data, then auto-saves once when a session
    finishes. Returns True if a data source is active.
    """
    # Must run whenever the port is open, not only once the device is
    # marked ready: device_ready is set BY reading the port, so gating
    # the read on it would deadlock.
    streaming = st.session_state.simulation or st.session_state.serial_connected

    if streaming:
        ingest()

    # Auto-save exactly once, the moment the goal-triggered stop happens
    # (the manual Stop button also guards on session_saved, so whichever
    # path fires first "wins" and the other is a no-op).
    if st.session_state.session_complete and not st.session_state.session_saved:
        save_session()
        st.session_state.session_saved = True

    return streaming


def keep_loop_alive(streaming):
    """Reruns the page while recording, so the display keeps updating."""
    if st.session_state.running:
        time.sleep(LOOP_SLEEP)
        st.rerun()
    elif streaming and not st.session_state.simulation:
        # Idle poll: drains boot banners, detects the device, keeps the
        # polarity check live before recording starts.
        time.sleep(IDLE_SLEEP)
        st.rerun()


def control_buttons(ready, labels):
    """Start / stop / reset row. `labels` lets each view word them its own way."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if not st.session_state.running:
            if st.button(
                labels["start"],
                type="primary",
                use_container_width=True,
                disabled=not (ready and st.session_state.calibrated),
            ):
                start_session()
                st.rerun()
        else:
            if st.button(labels["stop"], use_container_width=True):
                stop_session()
                if not st.session_state.session_saved:
                    save_session()
                    st.session_state.session_saved = True
                st.rerun()

    with col2:
        if st.button(labels["reset"], use_container_width=True):
            reset_measurements()
            st.rerun()

    return col3


def patient_finished_view(config):
    """
    What a patient sees after reaching the goal. Order matters:
    celebration first, then the two things they may want to do (add a
    note, move on), then the details. The attempt-by-attempt table is
    tucked away, since most patients only need the headline numbers.
    """
    completion_panel(config)

    notes_box("summary")

    st.markdown("#### How it went")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Attempts", st.session_state.repetitions)
    col2.metric("Correct", st.session_state.correct_repetitions)
    col3.metric("Not counted", st.session_state.incorrect_repetitions)
    col4.metric("Accuracy", f"{calculate_score():.0f}%")

    rep_history_chart()

    reps = repetition_dataframe()

    if not reps.empty:
        with st.expander("See every attempt"):
            st.dataframe(
                friendly_reps_table(
                    reps[["rep_number", "peak_angle", "duration_s", "result", "reason"]]
                ),
                use_container_width=True,
                hide_index=True,
                height=min(38 * (len(reps) + 1), 700),
            )

    session_downloads("summary")


def patient_live_page():
    """The patient view: guided steps, big coaching text, dots and gauge."""
    config = exercise_config()
    name = st.session_state.exercise
    style = EXERCISE_STYLE.get(name, EXERCISE_STYLE["Dorsiflexion"])

    st.markdown(
        f"""
        <div class="p-hero">
            <div class="txt">
                <h2>{style['emoji']} {name}</h2>
                <p>Aim for {config['target_min']}-{config['target_max']} degrees.
                   Complete {config['target_reps']} correct repetitions.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    finished = st.session_state.session_complete and st.session_state.goal_reached

    # The setup steps only matter before the goal is reached. Once the
    # patient has finished, the celebration is the screen, not a footnote
    # below a stale Stop button.
    if finished:
        streaming = run_acquisition()
        patient_finished_view(config)
        return

    patient_connection_panel()
    patient_connection_status()

    ready = st.session_state.serial_connected and st.session_state.device_ready

    # ---- step 1: neutral ----
    st.markdown("### Step 1 - Get comfortable")

    if not st.session_state.calibrated:
        patient_calibration_panel()
    else:
        st.success("Your neutral position is set.")

    # ---- step 2: exercise ----
    st.markdown("### Step 2 - Do the exercise")
    control_buttons(
        ready,
        {"start": "▶️ Start exercise", "stop": "⏹ Stop", "reset": "↺ Start over"},
    )

    was_calibrated = st.session_state.calibrated
    streaming = run_acquisition()

    # run_acquisition() can finish the session (auto-save) after the
    # controls above were drawn, so re-draw as the finished view.
    if st.session_state.session_complete and st.session_state.goal_reached:
        st.rerun()

    # The neutral is captured inside run_acquisition(), which runs AFTER
    # the buttons above were drawn. Without this rerun the screen would
    # show the old state (Start still greyed out) until something else
    # happened to trigger a refresh.
    if st.session_state.calibrated and not was_calibrated:
        st.rerun()

    # Until neutral is set there is nothing to coach: an empty gauge and
    # 15 blank dots would only distract from Step 1.
    if not st.session_state.calibrated:
        keep_loop_alive(streaming)
        return

    # ---- live coaching ----
    css_class, message = patient_instruction(config)
    st.markdown(
        f'<div class="instruction-card {css_class}">{message}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(rep_dots_html(config), unsafe_allow_html=True)
    st.markdown(angle_gauge_html(config), unsafe_allow_html=True)

    if st.session_state.last_rep_correct is True:
        st.success(
            f"Repetition counted! You reached {st.session_state.last_rep_peak:.0f} degrees."
        )
    elif st.session_state.last_rep_correct is False:
        st.warning(f"Not counted - {st.session_state.last_rep_reason}")

    if st.session_state.rep_records:
        st.subheader("Your repetitions")
        rep_history_chart()

    if st.session_state.session_complete:
        st.divider()
        st.info("You stopped before finishing. Here is how it went.")
        session_summary(config)

    keep_loop_alive(streaming)


def developer_live_page():
    """
    The developer view: the original technical layout, unchanged in what
    it shows and in what order. Device, calibration, raw numbers,
    polarity check, velocity trace and serial log.
    """
    config = exercise_config()

    st.markdown(
        '<div class="dev-banner"><b>LIVE / DEVELOPER</b> &nbsp;|&nbsp; '
        f'{st.session_state.exercise} &nbsp;|&nbsp; '
        f'source: {"simulation" if st.session_state.simulation else "hardware"}</div>',
        unsafe_allow_html=True,
    )

    connection_panel()
    st.divider()
    calibration_panel()
    st.divider()

    ready = st.session_state.simulation or st.session_state.device_ready

    state_col = control_buttons(
        ready,
        {"start": "Start recording", "stop": "Stop recording", "reset": "Reset"},
    )
    state_col.metric("State", st.session_state.movement_state)

    if not st.session_state.calibrated:
        st.info("Capture the neutral position to enable recording.")

    # Same ordering issue as the patient view: the neutral is captured
    # inside run_acquisition(), after the buttons were drawn.
    was_calibrated = st.session_state.calibrated
    streaming = run_acquisition()

    if st.session_state.calibrated and not was_calibrated:
        st.rerun()

    st.divider()
    st.subheader("Live measurements")

    # ---- completion takes over the page ----
    if st.session_state.session_complete and st.session_state.goal_reached:
        completion_panel(config)
        st.divider()
        session_summary(config)
        st.divider()
        st.subheader("Repetition history")
        rep_history_chart()

        with st.expander("Serial log"):
            st.code("\n".join(st.session_state.log_lines[-40:]) or "empty")

        return

    # ---- live numbers ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Angle", f"{st.session_state.current_angle:.1f} deg")
    col2.metric(
        "Correct reps",
        f"{st.session_state.correct_repetitions} / {config['target_reps']}",
    )
    col3.metric("Attempts", st.session_state.repetitions)
    col4.metric("Accuracy", f"{calculate_score():.0f}%")

    st.metric("Session ROM", f"{achieved_rom():.1f} deg")

    if st.session_state.last_rep_correct is not None:
        if st.session_state.last_rep_correct:
            st.success(
                f"Attempt {st.session_state.repetitions}: correct "
                f"(peak {st.session_state.last_rep_peak:.1f} deg). "
                f"{st.session_state.last_rep_reason}"
            )
        else:
            st.error(
                f"Attempt {st.session_state.repetitions}: not counted - "
                f"{st.session_state.last_rep_reason}"
            )

    progress = min(
        st.session_state.correct_repetitions / max(1, config["target_reps"]), 1.0
    )
    st.progress(progress)
    st.caption(
        f"{st.session_state.correct_repetitions} of {config['target_reps']} "
        f"correct repetitions  -  {st.session_state.repetitions} attempted, "
        f"{st.session_state.incorrect_repetitions} not counted"
    )

    polarity_panel()

    st.subheader("Ankle angle")
    live_chart()

    if st.session_state.rep_records:
        st.subheader("Repetition history")
        rep_history_chart()

    if st.session_state.session_complete:
        st.divider()
        st.info("Session stopped before the correct-rep goal was reached.")
        session_summary(config)

    with st.expander("Serial log"):
        st.code("\n".join(st.session_state.log_lines[-40:]) or "empty")

    keep_loop_alive(streaming)


def live_page():
    if is_developer():
        developer_live_page()
    else:
        patient_live_page()


# ============================================================
# HISTORY PAGE
#
# CHANGED: shows per-attempt detail for sessions recorded after the
# v3.0 upgrade. Older sessions have no per-attempt rows, because
# that data was never recorded at the time; they still appear in the
# session list exactly as before.
# ============================================================

def load_sessions():
    """All saved sessions for the current patient, newest first."""
    conn = sqlite3.connect(DB_NAME)
    frame = pd.read_sql_query(
        "SELECT * FROM sessions WHERE patient_id = ? ORDER BY date DESC, id DESC",
        conn,
        params=(st.session_state.patient_id,),
    )
    conn.close()
    return frame


def load_repetitions(session_id):
    conn = sqlite3.connect(DB_NAME)
    reps = pd.read_sql_query(
        "SELECT rep_number, peak_angle, duration_s, result, reason "
        "FROM repetitions WHERE session_id = ? ORDER BY rep_number",
        conn,
        params=(session_id,),
    )
    conn.close()
    return reps


def friendly_reps_table(reps):
    """Rename the stored codes into words a patient can read."""
    shown = reps.copy()
    shown["result"] = shown["result"].map(
        {"correct": "Correct", "incorrect": "Not counted"}
    )
    shown["reason"] = shown["reason"].map(
        {"ok": "Good range", "too_short": "Too short", "too_far": "Too far"}
    )
    shown.columns = ["Attempt", "Peak (deg)", "Time (s)", "Result", "Note"]
    return shown


def trend_png(frame):
    """
    Accuracy and range-of-motion over time, oldest to newest. Returned as
    PNG bytes so the chart can be cached instead of redrawn on every rerun.

    Accuracy (0-100%) and range of motion (typically 0-50 deg) share very
    different scales, so ROM is plotted on its own right-hand axis;
    otherwise it would look flat near the bottom of a 0-100 scale.
    """
    ordered = frame.iloc[::-1].reset_index(drop=True)
    x = list(range(1, len(ordered) + 1))

    figure, ax_acc = plt.subplots(figsize=(9, 3.2))
    ax_rom = ax_acc.twinx()

    line1, = ax_acc.plot(x, ordered["score"], marker="o", color="#0d9488",
                          linewidth=2.4, label="Accuracy (%)")
    line2, = ax_rom.plot(x, ordered["achieved_rom"], marker="s", color="#f97316",
                          linewidth=2.4, label="Range of motion (deg)")

    ax_acc.set_xlabel("Session number")
    ax_acc.set_ylabel("Accuracy (%)", color="#0d9488")
    ax_rom.set_ylabel("Range of motion (deg)", color="#f97316")
    ax_acc.set_ylim(0, 105)
    ax_acc.set_xticks(x if len(x) <= 15 else None)
    ax_acc.grid(True, alpha=0.25)
    ax_acc.legend(handles=[line1, line2], frameon=False, loc="lower right")

    for side in ("top",):
        ax_acc.spines[side].set_visible(False)
        ax_rom.spines[side].set_visible(False)

    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100, transparent=False, facecolor="white")
    plt.close(figure)
    return buffer.getvalue()


def exercise_breakdown(frame):
    """Per-exercise totals: sessions, correct reps, accuracy, best range."""
    rows = []

    for name in EXERCISES:
        part = frame[frame["exercise"] == name]

        if part.empty:
            rows.append((name, 0, 0, None, None))
            continue

        correct = int(part["correct_repetitions"].fillna(0).sum())
        attempted = int(part["repetitions"].fillna(0).sum())
        accuracy = round(100.0 * correct / attempted) if attempted else 0
        best = float(part["achieved_rom"].fillna(0).max())
        rows.append((name, len(part), correct, accuracy, best))

    return rows


def patient_history_page(frame):
    st.markdown("## 📈 My progress")

    total_sessions = len(frame)
    correct = int(frame["correct_repetitions"].fillna(0).sum())
    attempted = int(frame["repetitions"].fillna(0).sum())
    accuracy = round(100.0 * correct / attempted) if attempted else 0
    streak = practice_streak()

    st.markdown(
        f"""
        <div class="tiles">
            <div class="tile gold"><div class="v">🔥 {streak}</div><div class="l">day streak</div></div>
            <div class="tile teal"><div class="v">{total_sessions}</div><div class="l">{"session" if total_sessions == 1 else "sessions"}</div></div>
            <div class="tile coral"><div class="v">{correct}</div><div class="l">correct repetitions</div></div>
            <div class="tile blue"><div class="v">{accuracy}%</div><div class="l">overall accuracy</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- per exercise ----
    st.subheader("By exercise")

    cards = st.columns(len(EXERCISES))
    for column, (name, sessions, reps, acc, best) in zip(cards, exercise_breakdown(frame)):
        style = EXERCISE_STYLE[name]

        if sessions:
            plural = "session" if sessions == 1 else "sessions"
            detail = f"{sessions} {plural}<br>{acc}% accuracy<br>best {best:.0f} deg"
        else:
            detail = "Not started yet"

        with column:
            st.markdown(
                f"""
                <div class="ex-card {style['css']}">
                    <h4>{name}</h4>
                    <p>{detail}</p>
                    <span class="tag">{reps} correct reps</span>
                    <span class="emoji">{style['emoji']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---- trend ----
    st.subheader("How you're doing over time")

    if len(frame) < 2:
        st.info("Complete another session and your progress chart will appear here.")
    else:
        key = (len(frame), int(frame["id"].max()))

        if st.session_state.get("trend_key") != key:
            st.session_state.trend_png = trend_png(frame)
            st.session_state.trend_key = key

        st.image(st.session_state.trend_png, width="stretch")

    # ---- session cards ----
    st.subheader("Your sessions")

    for _, row in frame.head(10).iterrows():
        style = EXERCISE_STYLE.get(row["exercise"], EXERCISE_STYLE["Dorsiflexion"])
        when = str(row["date"])[:16]
        note = str(row["notes"]).strip() if pd.notna(row.get("notes")) else ""
        note_html = f'<div class="s-note">📝 {note}</div>' if note else ""

        st.markdown(
            f"""
            <div class="s-card">
                <div class="s-top">
                    <span class="s-name">{style['emoji']} {row['exercise']}</span>
                    <span class="s-when">{when}</span>
                </div>
                <div class="s-stats">
                    <span>✅ {int(row['correct_repetitions'] or 0)} correct</span>
                    <span>➖ {int(row['incorrect_repetitions'] or 0)} not counted</span>
                    <span>🎯 {float(row['score'] or 0):.0f}% accuracy</span>
                    <span>📏 {float(row['achieved_rom'] or 0):.0f} deg range</span>
                </div>
                {note_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if len(frame) > 10:
        st.caption(f"Showing your latest 10 of {len(frame)} sessions. Download the CSV for all of them.")

    st.download_button(
        "⬇️ Download all my sessions (CSV)",
        sessions_csv(patient_id=st.session_state.patient_id),
        file_name=f"{st.session_state.patient_id}_all_sessions.csv",
        mime="text/csv",
    )


def developer_history_page(frame):
    st.title("History")

    st.dataframe(
        frame.rename(columns={"score": "accuracy"}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Progress")

    if len(frame) >= 2:
        key = (len(frame), int(frame["id"].max()))

        if st.session_state.get("trend_key") != key:
            st.session_state.trend_png = trend_png(frame)
            st.session_state.trend_key = key

        st.image(st.session_state.trend_png, width="stretch")
    else:
        st.info("At least two sessions are needed for a trend.")

    st.subheader("Attempt-by-attempt detail")

    options = {
        f"Session {row['id']} - {row['exercise']} - {row['date']}": int(row["id"])
        for _, row in frame.iterrows()
    }

    choice = st.selectbox("Choose a session", list(options.keys()))
    reps = load_repetitions(options[choice])

    if reps.empty:
        st.info(
            "No attempt-by-attempt detail for this session. It was "
            "recorded before per-attempt storage was added."
        )
    else:
        st.dataframe(friendly_reps_table(reps), use_container_width=True, hide_index=True)

    st.download_button(
        "Download all sessions with notes (CSV)",
        sessions_csv(patient_id=st.session_state.patient_id),
        file_name=f"{st.session_state.patient_id}_all_sessions.csv",
        mime="text/csv",
    )


def history_page():
    frame = load_sessions()

    if frame.empty:
        if is_developer():
            st.title("History")
            st.info("No sessions recorded for this patient yet.")
        else:
            st.markdown("## 📈 My progress")
            st.info(
                "Your progress will show up here after your first session. "
                "Pick an exercise on Home to get started."
            )
        return

    if is_developer():
        developer_history_page(frame)
    else:
        patient_history_page(frame)


# ============================================================
# SETTINGS PAGE  (developer only)
# ============================================================

def settings_page():
    st.title("Settings")

    st.subheader("Axis and sign per exercise")
    st.caption(
        "The firmware transmits the roll of the neutral-corrected relative "
        "orientation as its Ankle value, which is the number validated "
        "against Kinovea - keep 'firmware' for dorsiflexion and "
        "plantarflexion. Inversion and eversion have to be read from "
        "another axis of the reconstructed orientation; determine which "
        "one experimentally."
    )

    for name in EXERCISES:
        config = exercise_config(name)

        col1, col2, col3 = st.columns([2, 2, 2])
        col1.write(f"**{name}**")

        axis = col2.selectbox(
            "Axis",
            AXIS_CHOICES,
            index=AXIS_CHOICES.index(config["axis"]),
            key=f"axis_{name}",
            label_visibility="collapsed",
        )

        sign = col3.selectbox(
            "Direction",
            ["Positive", "Negative"],
            index=0 if config["sign"] > 0 else 1,
            key=f"sign_{name}",
            label_visibility="collapsed",
        )

        st.session_state.axis_overrides[name] = axis
        st.session_state.sign_overrides[name] = 1 if sign == "Positive" else -1

    st.divider()

    st.subheader("Repetition detection")

    st.session_state.min_peak_attempt = st.slider(
        "Smallest movement recorded as an attempt (degrees)",
        1.0, 10.0, float(st.session_state.min_peak_attempt), 0.5,
        help="Movements whose peak stays below this are treated as sensor "
             "noise and ignored. Anything at or above it is recorded, and "
             "marked incorrect if it does not reach the target range.",
    )

    st.session_state.return_threshold = st.slider(
        "Return-to-neutral threshold (degrees)",
        1.0, 15.0, float(st.session_state.return_threshold), 0.5,
        help="The angle below which the foot is considered back at "
             "neutral, closing the repetition.",
    )

    st.caption(
        "A movement starts an attempt once it rises above the return "
        f"threshold plus one degree (currently "
        f"{st.session_state.return_threshold + 1.0:.1f} deg), so any "
        "tracked attempt already clears the return threshold on its own. "
        "The smallest-attempt value only matters if it is set HIGHER than "
        "that starting point - use it to also discard small but real "
        "movements, not to protect against resting drift, which the "
        "return threshold already does."
    )

    entry_point = st.session_state.return_threshold + 1.0

    if st.session_state.min_peak_attempt > entry_point + 8.0:
        st.warning(
            f"The smallest-attempt value ({st.session_state.min_peak_attempt:.1f} deg) "
            f"is set well above where attempts start being tracked "
            f"({entry_point:.1f} deg). Genuine small movements within the "
            "target range could be discarded. Consider lowering it."
        )

    st.divider()

    st.subheader("Yaw drift")
    st.caption(
        "The firmware runs Madgwick on accelerometer and gyroscope only - "
        "the magnetometer is not used. Roll and pitch are gravity-"
        "referenced and stable; yaw has no absolute reference and will "
        "drift. Prefer roll or pitch for every exercise, and report the "
        "drift over a five-minute static hold as a known limitation."
    )

    st.divider()

    st.subheader("Signal chain")
    st.code(
        "MPU9250 foot (0x68) + MPU9250 shank (0x69)\n"
        "  -> ESP32 I2C at 100 Hz\n"
        "  -> Madgwick fusion, beta = 0.08\n"
        "  -> qFoot, qShank\n"
        "  -> qRelative = conj(qShank) * qFoot\n"
        "  -> qMotion   = conj(qNeutral) * qRelative\n"
        "  -> Serial text at 115200 baud\n"
        "  -> Python parser -> Euler to quaternion reconstruction\n"
        "  -> axis selection per exercise\n"
        "  -> ROM, repetition counting, correctness grading\n"
        "  -> Streamlit GUI -> SQLite + CSV"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    # One theme for the whole session, chosen from the current mode.
    inject_style()

    if not st.session_state.logged_in:
        login_page()
        return

    sidebar()

    pages = {
        "Dashboard": dashboard_page,
        "Exercise": exercise_page,
        "Live": live_page,
        "History": history_page,
        "Settings": settings_page,
    }

    page = st.session_state.page

    # Settings and the standalone Exercise page are developer-only now that
    # exercise selection, instructions and video live on the patient Home
    # page; fall back if a patient somehow lands on either.
    if page in ("Settings", "Exercise") and not is_developer():
        page = "Dashboard"

    pages.get(page, dashboard_page)()


if __name__ == "__main__":
    main()