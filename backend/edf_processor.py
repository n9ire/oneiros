"""
EDF / MNE-Python biosignal processing for Oneiros.

Provides:
  load_edf(path)            → metadata + preview waveform
  apply_edf_pipeline(...)   → processed epochs as CustomDatasetPayload
  generate_edf_plot(...)    → PNG bytes (waveform or PSD)

Requires:  pip install mne
"""

from __future__ import annotations

import base64
import io
import math
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

# MNE import — we defer the heavy import so the server still starts if MNE is
# not installed (endpoints will return a 503 with an install hint).
def _mne():
    try:
        import mne
        return mne
    except ImportError:
        raise RuntimeError("MNE-Python is not installed. Run: pip install mne")


# ── Temp session store ────────────────────────────────────────────────────────
# Maps session_id → (raw_data: np.ndarray, info_dict: dict)
# raw_data shape: (n_channels, n_times)
_sessions: dict[str, dict[str, Any]] = {}


# ── Load ──────────────────────────────────────────────────────────────────────

def load_edf(path: str | Path) -> dict:
    """
    Load an EDF file with MNE and return metadata + a short preview array.

    Returns
    -------
    {
      sessionId: str,
      name: str,
      channels: list[str],
      sfreq: float,
      duration: float,          # seconds
      nTimes: int,
      events: list[{id, label, count}],
      previewData: list[list[float]],   # (n_channels, preview_samples) — 10 s max
      previewSfreq: float,              # sfreq of previewData (downsampled if needed)
    }
    """
    mne = _mne()
    path = Path(path)

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    info = raw.info

    channels: list[str] = list(raw.ch_names)
    sfreq: float = float(info["sfreq"])
    n_times: int = int(raw.n_times)
    duration: float = n_times / sfreq

    # Events from annotations
    events_out: list[dict] = []
    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        counts: dict[int, int] = {}
        for ev in events:
            counts[int(ev[2])] = counts.get(int(ev[2]), 0) + 1
        id_to_label = {v: k for k, v in event_id.items()}
        for eid, cnt in counts.items():
            events_out.append({"id": eid, "label": id_to_label.get(eid, str(eid)), "count": cnt})
    except Exception:
        pass

    # Preview: at most 10 s, downsampled to ≤256 Hz for transfer efficiency
    preview_sfreq = min(sfreq, 256.0)
    preview_secs = min(10.0, duration)
    preview_samples = int(preview_secs * preview_sfreq)

    if preview_sfreq < sfreq:
        raw_preview = raw.copy().resample(preview_sfreq, verbose=False)
    else:
        raw_preview = raw

    data_preview, _ = raw_preview[:, : int(preview_secs * preview_sfreq)]
    preview_data = data_preview.tolist()  # list[list[float]]

    # Store full raw for later pipeline processing
    session_id = str(uuid.uuid4())
    raw_data, _ = raw[:, :]
    _sessions[session_id] = {
        "raw_data": raw_data,          # (n_channels, n_times)
        "ch_names": channels,
        "sfreq": sfreq,
        "n_times": n_times,
        "path": str(path),
        "name": path.stem,
        "raw": raw,                    # keep the MNE Raw object
    }

    return {
        "sessionId": session_id,
        "name": path.stem,
        "channels": channels,
        "sfreq": sfreq,
        "duration": duration,
        "nTimes": n_times,
        "events": events_out,
        "previewData": preview_data,
        "previewSfreq": preview_sfreq,
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────

def apply_edf_pipeline(session_id: str, steps: list[dict]) -> dict:
    """
    Apply an ordered list of EDF pipeline steps to a loaded session.

    Each step is {type: str, **params}.

    Supported steps
    ---------------
    bandpass       : {"lo": float, "hi": float}
    notch          : {"freq": float}   (50 or 60 Hz)
    resample       : {"sfreq": float}
    pick_channels  : {"channels": list[str]}
    epoch          : {"mode": "fixed"|"event", "window": float, "overlap": float,
                       "eventId": int|None, "tmin": float, "tmax": float}
    ica            : {"nComponents": int, "eogChannel": str|None}

    Returns
    -------
    CustomDatasetPayload-compatible dict:
    {
      X_train, y_train, X_val, y_val,
      featureCount, classCount, featureNames, classNames,
      datasetName, trainSamples, valSamples
    }
    """
    mne = _mne()

    if session_id not in _sessions:
        raise ValueError(f"Session {session_id!r} not found. Upload the EDF again.")

    sess = _sessions[session_id]
    # Work on a copy so repeated calls don't corrupt the session
    raw: Any = sess["raw"].copy()

    for step in steps:
        stype = step.get("type", "")

        if stype == "bandpass":
            lo = float(step.get("lo", 0.5))
            hi = float(step.get("hi", 40.0))
            raw.filter(lo, hi, verbose=False)

        elif stype == "notch":
            freq = float(step.get("freq", 50.0))
            raw.notch_filter(freq, verbose=False)

        elif stype == "resample":
            target = float(step.get("sfreq", 128.0))
            if target < raw.info["sfreq"]:
                raw.resample(target, verbose=False)

        elif stype == "pick_channels":
            channels = step.get("channels", [])
            if channels:
                available = [c for c in channels if c in raw.ch_names]
                if available:
                    raw.pick_channels(available, verbose=False)

        elif stype == "ica":
            n_components = int(step.get("nComponents", min(15, len(raw.ch_names))))
            eog_ch = step.get("eogChannel", None)
            ica = mne.preprocessing.ICA(
                n_components=n_components,
                random_state=42,
                verbose=False,
            )
            ica.fit(raw, verbose=False)
            if eog_ch and eog_ch in raw.ch_names:
                eog_inds, _ = ica.find_bads_eog(raw, ch_name=eog_ch, verbose=False)
                ica.exclude = eog_inds
            ica.apply(raw, verbose=False)

    # Epoch step (separate pass so it comes last)
    epoch_step = next((s for s in steps if s.get("type") == "epoch"), None)

    if epoch_step is None:
        # No epoch step — use fixed 1-s windows with no labels
        window = 1.0
        overlap = 0.0
        X, y, class_names = _fixed_epochs(raw, window, overlap)
    elif epoch_step.get("mode") == "event":
        event_id_target = epoch_step.get("eventId", None)
        tmin = float(epoch_step.get("tmin", -0.2))
        tmax = float(epoch_step.get("tmax", 0.8))
        X, y, class_names = _event_epochs(mne, raw, event_id_target, tmin, tmax)
    else:
        window  = float(epoch_step.get("window", 1.0))
        overlap = float(epoch_step.get("overlap", 0.0))
        X, y, class_names = _fixed_epochs(raw, window, overlap)

    if len(X) == 0:
        raise ValueError("No epochs produced. Check filter / epoch settings.")

    feature_names = [f"{ch}_t{i}" for ch in raw.ch_names for i in range(X.shape[2])]

    # Flatten epochs: (n_epochs, n_channels, n_times) → (n_epochs, n_channels * n_times)
    n_epochs = X.shape[0]
    X_flat = X.reshape(n_epochs, -1).tolist()
    y_list  = y.tolist()

    # 80/20 train/val split
    split   = max(1, int(n_epochs * 0.8))
    X_train = X_flat[:split];  y_train = y_list[:split]
    X_val   = X_flat[split:];  y_val   = y_list[split:]

    if not X_val:
        X_val  = X_flat[:1]
        y_val  = y_list[:1]

    return {
        "X_train":     X_train,
        "y_train":     y_train,
        "X_val":       X_val,
        "y_val":       y_val,
        "featureCount": len(feature_names),
        "classCount":   len(class_names),
        "featureNames": feature_names,
        "classNames":   class_names,
        "datasetName":  sess["name"],
        "trainSamples": len(X_train),
        "valSamples":   len(X_val),
    }


def _fixed_epochs(
    raw: Any,
    window: float,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sliding-window epochs, single class (label 0)."""
    sfreq = raw.info["sfreq"]
    data, _ = raw[:, :]
    n_ch, n_t = data.shape
    win_samples = int(window * sfreq)
    step = int((window - overlap) * sfreq)
    if step <= 0:
        step = win_samples

    epochs = []
    starts = range(0, n_t - win_samples + 1, step)
    for s in starts:
        epochs.append(data[:, s : s + win_samples])

    if not epochs:
        return np.empty((0, n_ch, win_samples)), np.empty(0, dtype=int), ["window"]

    X = np.stack(epochs, axis=0)  # (n_epochs, n_ch, n_times)
    y = np.zeros(len(epochs), dtype=int)
    return X, y, ["window"]


def _event_epochs(
    mne: Any,
    raw: Any,
    event_id_target: int | None,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Event-locked epochs."""
    try:
        events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    except Exception:
        events = np.empty((0, 3), dtype=int)
        event_id_map = {}

    if len(events) == 0:
        raise ValueError("No events found in EDF annotations. Switch to fixed-window epoching.")

    if event_id_target is not None:
        id_to_label = {v: k for k, v in event_id_map.items()}
        sel_label = id_to_label.get(event_id_target)
        event_id_sel = {sel_label: event_id_target} if sel_label else event_id_map
    else:
        event_id_sel = event_id_map

    epochs_mne = mne.Epochs(
        raw,
        events,
        event_id=event_id_sel,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )
    X = epochs_mne.get_data()                          # (n_epochs, n_ch, n_times)
    events_used = epochs_mne.events[:, 2]
    id_to_label = {v: k for k, v in event_id_map.items()}
    labels = sorted(set(int(e) for e in events_used))
    label_idx = {eid: i for i, eid in enumerate(labels)}
    y = np.array([label_idx[int(e)] for e in events_used], dtype=int)
    class_names = [id_to_label.get(eid, str(eid)) for eid in labels]
    return X, y, class_names


# ── Visualisation ─────────────────────────────────────────────────────────────

def generate_edf_plot(session_id: str, config: dict) -> bytes:
    """
    Generate a PNG visualisation for an EDF session.

    config keys
    -----------
    plot_type : "waveform" | "psd"
    channels  : list[str]  (empty = first 8)
    duration  : float      (waveform: seconds to show, default 5)
    fmin      : float      (psd: min Hz, default 0)
    fmax      : float      (psd: max Hz, default 80)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    DARK = "#09090b"
    PANEL = "#18181b"
    TEXT  = "#d4d4d8"
    GRID  = "#27272a"
    COLORS = ["#6366f1", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444",
              "#a855f7", "#ec4899", "#14b8a6"]

    if session_id not in _sessions:
        raise ValueError(f"Session {session_id!r} not found.")

    sess = _sessions[session_id]
    raw = sess["raw"]

    plot_type = config.get("plot_type", "waveform")
    ch_names  = config.get("channels", []) or raw.ch_names[:8]
    ch_names  = [c for c in ch_names if c in raw.ch_names] or raw.ch_names[:8]

    fig = plt.figure(figsize=(12, max(3, len(ch_names) * 0.9 + 1)), facecolor=DARK)

    if plot_type == "psd":
        fmin = float(config.get("fmin", 0))
        fmax = float(config.get("fmax", 80))
        mne = _mne()
        raw_sel = raw.copy().pick_channels(ch_names, verbose=False)
        spectrum = raw_sel.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
        psds, freqs = spectrum.get_data(return_freqs=True)

        ax = fig.add_subplot(111, facecolor=PANEL)
        for i, (ch, psd) in enumerate(zip(ch_names, psds)):
            psd_db = 10 * np.log10(psd + 1e-30)
            ax.plot(freqs, psd_db, color=COLORS[i % len(COLORS)], linewidth=1.2, label=ch, alpha=0.85)
        ax.set_xlabel("Frequency (Hz)", color=TEXT, fontsize=9)
        ax.set_ylabel("Power (dB)", color=TEXT, fontsize=9)
        ax.set_title("Power Spectral Density", color=TEXT, fontsize=10, pad=8)
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, loc="upper right",
                  ncol=min(4, len(ch_names)), framealpha=0.6)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.grid(color=GRID, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    else:  # waveform
        duration = float(config.get("duration", 5.0))
        raw_sel = raw.copy().pick_channels(ch_names, verbose=False)
        sfreq = raw_sel.info["sfreq"]
        n_show = int(min(duration, raw_sel.n_times / sfreq) * sfreq)
        data, times = raw_sel[:, :n_show]

        gs = gridspec.GridSpec(len(ch_names), 1, hspace=0.05)
        axes = [fig.add_subplot(gs[i]) for i in range(len(ch_names))]

        for i, (ch, ax) in enumerate(zip(ch_names, axes)):
            row = data[i]
            ax.set_facecolor(PANEL)
            ax.plot(times, row, color=COLORS[i % len(COLORS)], linewidth=0.7)
            ax.set_ylabel(ch, color=TEXT, fontsize=7, rotation=0,
                          ha="right", va="center", labelpad=40)
            ax.tick_params(left=False, labelleft=False, colors=TEXT, labelsize=7)
            ax.grid(axis="x", color=GRID, linewidth=0.4)
            for spine in ax.spines.values():
                spine.set_color(GRID)
            if i < len(ch_names) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Time (s)", color=TEXT, fontsize=9)

        fig.suptitle(f"EEG Waveform — {sess['name']}", color=TEXT, fontsize=10, y=1.01)

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=DARK, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Meta (step schemas) ───────────────────────────────────────────────────────

EDF_STEP_META = [
    {
        "type": "bandpass",
        "label": "Bandpass Filter",
        "params": [
            {"key": "lo",  "label": "Low (Hz)",  "type": "number", "default": 0.5},
            {"key": "hi",  "label": "High (Hz)", "type": "number", "default": 40.0},
        ],
    },
    {
        "type": "notch",
        "label": "Notch Filter",
        "params": [
            {"key": "freq", "label": "Freq (Hz)", "type": "select",
             "options": [50, 60], "default": 50},
        ],
    },
    {
        "type": "resample",
        "label": "Resample",
        "params": [
            {"key": "sfreq", "label": "Target Hz", "type": "number", "default": 128},
        ],
    },
    {
        "type": "pick_channels",
        "label": "Pick Channels",
        "params": [
            {"key": "channels", "label": "Channels (comma-sep)", "type": "text", "default": ""},
        ],
    },
    {
        "type": "epoch",
        "label": "Epoch",
        "params": [
            {"key": "mode",    "label": "Mode",          "type": "select",  "options": ["fixed", "event"], "default": "fixed"},
            {"key": "window",  "label": "Window (s)",    "type": "number",  "default": 1.0},
            {"key": "overlap", "label": "Overlap (s)",   "type": "number",  "default": 0.0},
            {"key": "tmin",    "label": "tmin (s)",      "type": "number",  "default": -0.2},
            {"key": "tmax",    "label": "tmax (s)",      "type": "number",  "default": 0.8},
            {"key": "eventId", "label": "Event ID",      "type": "number",  "default": None},
        ],
    },
    {
        "type": "ica",
        "label": "ICA Artefact Removal",
        "params": [
            {"key": "nComponents",  "label": "Components",   "type": "number", "default": 15},
            {"key": "eogChannel",   "label": "EOG Channel",  "type": "text",   "default": ""},
        ],
    },
]
