"""Skin-lesion analysis: upload -> U-Net lesion mask -> melanoma classifier (MelaNet) -> dashboard.

Run:  python app.py   (then open http://<server-ip>:7860 from a laptop or phone on the same Wi-Fi)
Research prototype only: not a medical device, not a diagnosis.
"""
import json
import os
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps

import classifier

# ----------------------------------------------------------------- config
IMG_SIZE = 256  # must match the input size your U-Net was trained with
UNET_PATH = Path("weights/unet.keras")         # your U-Net
SEG_JSON, CLS_JSON = Path("metrics/seg_metrics.json"), Path("metrics/cls_metrics.json")
DEFAULT_THRESHOLD = 0.5   # P(malignant) >= threshold -> 'malignant'; adjustable in the UI

# ----------------------------------------------------------------- models
def _load_unet(path):
    # compile=False: inference only, so the custom composite loss/metrics are not needed.
    # If your model uses custom *layers*, pass custom_objects={"LayerName": LayerClass}.
    try:
        return tf.keras.models.load_model(path, compile=False)
    except (OSError, ValueError) as e:
        print(f"[not loaded] {path}: {e}")
        return None


unet = _load_unet(UNET_PATH)  # None -> demo mask, so the network/dashboard parts can still be tested
clf = None
try:
    clf = classifier.Classifier()
except Exception as e:  # offline (first run downloads the weights), corrupt file ...
    print(f"[classifier not loaded] {e}")
clf_ft = None  # MelaNet fine-tuned on U-Net lesion crops (finetune_classifier.py); used for the crop if it exists
if clf is not None and classifier.FINETUNED.exists():
    try:
        clf_ft = classifier.Classifier(classifier.FINETUNED)
    except Exception as e:
        print(f"[fine-tuned classifier not loaded] {e}")


def _pad_to_square(img: np.ndarray):
    """Pads HxWx3 image to a square (S, S, 3) and returns (padded_img, (top, left, h, w, s))."""
    h, w = img.shape[:2]
    s = max(h, w)
    top = (s - h) // 2
    left = (s - w) // 2
    sq = np.zeros((s, s, img.shape[2]), dtype=img.dtype)
    sq[top:top + h, left:left + w] = img
    return sq, (top, left, h, w, s)


def _unpad_mask(mask_256: np.ndarray, pad_info) -> np.ndarray:
    """Resizes 256x256 square mask back to original (H, W) bool mask without distortion."""
    top, left, h, w, s = pad_info
    mask_u8 = (mask_256.astype(np.uint8)) * 255
    sq_mask = Image.fromarray(mask_u8, mode="L").resize((s, s), resample=Image.Resampling.NEAREST)
    orig_mask = sq_mask.crop((left, top, left + w, top + h))
    return np.array(orig_mask) > 127


def analyse(img: np.ndarray, use_mask=True, classify=True):
    """img: HxWx3 uint8 -> original-resolution bool mask (HxW), P(malignant) on original & lesion crop."""
    t0 = time.perf_counter()
    img = img[..., :3]
    sq_img, pad_info = _pad_to_square(img)
    x = tf.image.resize(sq_img, (IMG_SIZE, IMG_SIZE)).numpy()[None] / 255.0  # (1, 256, 256, 3)

    if unet is None:  # demo stub so the client-server stack can be tested before the model exists
        mask_256 = (x.mean(-1, keepdims=True) < x.mean())[0, ..., 0].numpy()
    else:
        mask_256 = (unet(x, training=False).numpy() > 0.5)[0, ..., 0]

    m = _unpad_mask(mask_256, pad_info)  # exact (H, W) matching original image
    t1 = time.perf_counter()

    out = {"mask": m, "p_orig": None, "p_mask": None}
    if clf is not None and classify:
        pil = Image.fromarray(img)
        out["p_orig"] = clf(pil)
        if use_mask and m.any():  # crop original image using exact-resolution mask
            out["p_mask"] = (clf_ft or clf)(classifier.crop_lesion(pil, m))

    out["ms_seg"] = 1000 * (t1 - t0)
    out["ms_cls"] = 1000 * (time.perf_counter() - t1)
    return out


def run_pipeline(img: np.ndarray, threshold=DEFAULT_THRESHOLD, use_mask=True):
    """Returns full-res mask, full-res overlay, decision text (original), decision text (lesion crop), details table."""
    r = analyse(img, use_mask)
    m = r["mask"]
    overlay = img[..., :3].copy()
    if m.any():
        overlay[m] = (0.5 * overlay[m] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)

    def show(p):
        if clf is None:
            return "classifier not loaded"
        if p is None:
            return "not requested"
        return f"{'MALIGNANT' if p >= threshold else 'benign'}   (P(malignant) = {p:.3f}, threshold {threshold})"

    details = pd.DataFrame([
        ("Lesion area (% of image)", round(100 * m.mean(), 2)),
        ("Segmentation time (ms)", round(r["ms_seg"], 1)),
        ("Classification time (ms)", round(r["ms_cls"], 1)),
        ("Lesion-crop classifier", "fine-tuned MelaNet" if clf_ft else "MelaNet (not fine-tuned)"),
    ], columns=["Metric", "Value"]).astype(str)
    return (m.astype(np.uint8) * 255), overlay, show(r["p_orig"]), show(r["p_mask"]), details

# ----------------------------------------------------------------- request log (network metrics)
# One row per request: (timestamp, latency_s, success, client_ip, upload_bytes). Kept for the whole run.
LOG, LOCK = [], threading.Lock()
STATE = {"active": 0, "peak": 0}   # requests currently in flight / highest seen at once
START = time.time()


def predict(img_path, threshold, use_mask, request: gr.Request):
    """API endpoint: called by the browser UI or by any gradio_client."""
    t0, ok, size = time.perf_counter(), False, 0
    with LOCK:
        STATE["active"] += 1
        STATE["peak"] = max(STATE["peak"], STATE["active"])
    try:
        if img_path is None:
            raise gr.Error("Upload a skin-lesion image first.")
        size = os.path.getsize(img_path)  # real bytes received over the network
        img = np.array(ImageOps.exif_transpose(Image.open(img_path)).convert("RGB"))
        out = run_pipeline(img, threshold, use_mask)
        ok = True
        return out
    finally:
        with LOCK:
            STATE["active"] -= 1
            LOG.append((time.time(), time.perf_counter() - t0, ok, request.client.host if request else "?", size))


def reset_stats():
    global START
    with LOCK:
        LOG.clear()
        STATE["peak"] = STATE["active"]
        START = time.time()


def network_view():
    now = time.time()
    with LOCK:
        df = pd.DataFrame(LOG, columns=["t", "latency", "ok", "client", "bytes"]).astype(
            {"t": float, "latency": float, "ok": bool, "client": str, "bytes": float})
        active, peak = STATE["active"], STATE["peak"]
    n, good = len(df), df[df.ok]
    s = (good.latency * 1000).describe(percentiles=[.5, .95, .99]).fillna(0)  # latency of successful requests
    busy = df.t.max() - df.t.min() if n > 1 else 0                            # first -> last request
    r = lambda v, d=1: round(float(np.nan_to_num(v)), d)

    rows = [
        ("Reliability", "Total requests", n),
        ("Reliability", "Failed requests", n - len(good)),
        ("Reliability", "Success rate (%)", r(100 * len(good) / n) if n else 0),
        ("Latency", "Average (ms)", r(s["mean"])),
        ("Latency", "Median (ms)", r(s["50%"])),
        ("Latency", "P95 (ms)", r(s["95%"])),
        ("Latency", "P99 (ms)", r(s["99%"])),
        ("Latency", "Min (ms)", r(s["min"])),
        ("Latency", "Max (ms)", r(s["max"])),
        ("Latency", "Jitter, std dev (ms)", r(s["std"])),
        ("Throughput", "Average (req/s)", r((n - 1) / busy, 2) if busy > 0 else 0),
        ("Throughput", "Peak, best 1 s (req/s)", int(df.t.astype(int).value_counts().max()) if n else 0),
        ("Clients", "Active (last 60 s)", df[df.t > now - 60].client.nunique()),
        ("Clients", "Total unique", df.client.nunique()),
        ("Load", "In-flight requests now", active),
        ("Load", "Peak concurrent requests", peak),
        ("Data", "Total received (MB)", r(df.bytes.sum() / 1e6, 2)),
        ("Data", "Avg upload size (KB)", r(df.bytes[df.bytes > 0].mean() / 1e3)),
        ("Server", "Uptime (min)", r((now - START) / 60)),
    ]
    summary = pd.DataFrame([(g, k, str(v)) for g, k, v in rows], columns=["Group", "Metric", "Value"])

    by = df.groupby("client").agg(Requests=("ok", "size"), Success=("ok", "mean"), Data=("bytes", "sum"))
    clients = pd.DataFrame({"Client IP": by.index, "Requests": by.Requests,
                            "Success rate (%)": (by.Success * 100).round(1),
                            "Avg latency (ms)": (good.groupby("client").latency.mean().reindex(by.index) * 1000).round(1).fillna(0),
                            "Data (KB)": (by.Data / 1e3).round(1)})

    plot = pd.DataFrame({"request": range(1, n + 1), "latency_ms": df.latency * 1000}).tail(200)
    return summary, clients, plot


# ----------------------------------------------------------------- dashboard (static metrics from offline evaluation)
def _json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def static_metrics():
    table = lambda p: pd.DataFrame(list(_json(p).items()), columns=["Metric", "Value"])
    return table(SEG_JSON), table(CLS_JSON)


# ----------------------------------------------------------------- UI
with gr.Blocks(title="Skin Lesion Analysis") as demo:
    gr.Markdown("# Skin lesion segmentation and melanoma classification")

    with gr.Tab("Analyse"):
        gr.Markdown("Research prototype for a student project. Not a medical device; the output is not a diagnosis.")
        with gr.Row():
            inp = gr.Image(type="filepath", label="Skin-lesion image", sources=["upload", "webcam"])
            with gr.Column():
                op = gr.Slider(0.05, 0.95, value=DEFAULT_THRESHOLD, step=0.01, label="Decision threshold: malignant if P(malignant) is at least this")
                use_mask = gr.Checkbox(True, label="Also classify the U-Net lesion crop (shows what the segmentation adds)")
                details = gr.Dataframe(label="Details")
        with gr.Row():
            mask_out = gr.Image(label="Lesion mask")
            overlay_out = gr.Image(label="Overlay")
        with gr.Row():
            label_orig = gr.Textbox(label="Classifier: original image", interactive=False)
            label_mask = gr.Textbox(label="Classifier: U-Net lesion crop", interactive=False)
        gr.Button("Analyse", variant="primary").click(
            predict, [inp, op, use_mask], [mask_out, overlay_out, label_orig, label_mask, details], api_name="predict")

    with gr.Tab("Network"):
        gr.Markdown("### Network metrics (cumulative since server start or last reset)")
        net_df = gr.Dataframe()
        gr.Markdown("**Per client**")
        client_df = gr.Dataframe()
        lat_plot = gr.LinePlot(x="request", y="latency_ms", title="Server latency, last 200 requests (ms)")
        reset = gr.Button("Reset network statistics")

        net_out = [net_df, client_df, lat_plot]
        demo.load(network_view, None, net_out)
        gr.Timer(2).tick(network_view, None, net_out)   # redraws the table; values are cumulative, not windowed
        reset.click(reset_stats).then(network_view, None, net_out)

    with gr.Tab("Evaluation"):
        gr.Markdown("### Segmentation metrics")
        seg_df = gr.Dataframe()
        gr.Markdown("### Classification metrics (original image vs U-Net lesion variants)")
        cls_df = gr.Dataframe()
        demo.load(static_metrics, None, [seg_df, cls_df])

if __name__ == "__main__":
    # 0.0.0.0 makes the server reachable from other devices on the same Wi-Fi
    demo.queue(default_concurrency_limit=2).launch(server_name="0.0.0.0", server_port=7860, show_error=True)
