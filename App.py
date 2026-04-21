"""
SOFTDESBG6 — Pothole Detection API
YOLO + Supabase DB + Storage + Daily Reports + Email Alerts

Email options (set in .env):
  Gmail App Password  : SMTP_HOST=smtp.gmail.com  SMTP_PORT=465  SSL=true
  Brevo (free tier)  : SMTP_HOST=smtp-relay.brevo.com  SMTP_PORT=587  SSL=false
  Outlook/Hotmail    : SMTP_HOST=smtp.office365.com  SMTP_PORT=587  SSL=false
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import uuid
import json
import csv
import io
import os
import base64
import smtplib
import tempfile
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta, timezone, date
from dotenv import load_dotenv

load_dotenv()

# ── Supabase ──────────────────────────────────────────────────
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
supabase: Client = None

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"✅ Supabase connected: {SUPABASE_URL}")
except Exception as e:
    print(f"⚠️  Supabase failed: {e}")

# ── YOLO ──────────────────────────────────────────────────────
from ultralytics import YOLO

WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "best.pt")
model   = None
try:
    model = YOLO(WEIGHTS)
    print(f"✅ Loaded: {WEIGHTS}")
except Exception as e:
    print(f"⚠️  YOLO load failed: {e}")

# ── Flask ──────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5173", "http://localhost:5174",
    "http://127.0.0.1:5173", "http://127.0.0.1:5174"
])

BUCKET = "pothole-media"


# ── Helpers ────────────────────────────────────────────────────

def upload_storage(file_bytes: bytes, filename: str, folder="images") -> str:
    if not supabase:
        return ""
    try:
        path = f"{folder}/{filename}"
        supabase.storage.from_(BUCKET).upload(
            path, file_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )
        # Build public URL manually — avoids get_public_url() format issues
        # across different supabase-py versions
        url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{path}"
        print(f"✅ Uploaded to storage: {url}")
        return url
    except Exception as e:
        print(f"⚠️  Storage upload failed: {e}")
        return ""


def run_yolo(frame: np.ndarray) -> list:
    if model is None:
        return []
    try:
        h, w    = frame.shape[:2]
        results = model(frame, verbose=False)
        out     = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf   = float(box.conf[0])
                bw, bh = x2 - x1, y2 - y1
                ratio  = (bw * bh) / (h * w)
                sev    = "High" if ratio > 0.04 else ("Medium" if ratio > 0.015 else "Low")
                out.append({
                    "id":         str(uuid.uuid4()),
                    "bbox":       [int(x1), int(y1), int(bw), int(bh)],
                    "confidence": round(conf * 100, 1),
                    "severity":   sev,
                    "timestamp":  datetime.now().isoformat()
                })
        return out
    except Exception as e:
        print(f"⚠️  YOLO error: {e}")
        return []


def save_detection(det: dict, source: str, image_url="", video_url="",
                   barangay="", frame_number=None, location_label=""):
    if not supabase:
        return
    try:
        supabase.table("pothole_detections").insert({
            "severity":       det.get("severity", "Low"),
            "confidence":     det.get("confidence", 0),
            "bbox":           det.get("bbox", []),
            "source":         source,
            "image_url":      image_url,
            "video_url":      video_url,
            "barangay":       barangay or None,
            "frame_number":   frame_number,
            "location_label": location_label or None,
            "detected_at":    det.get("timestamp", datetime.now().isoformat()),
        }).execute()
    except Exception as e:
        print(f"⚠️  DB save failed: {e}")


def update_daily_report(detections: list, barangay="", source=""):
    """Update today's daily report aggregate."""
    if not supabase or not detections:
        return
    try:
        today = date.today().isoformat()

        existing = supabase.table("daily_reports") \
            .select("*").eq("report_date", today).execute()

        if existing.data:
            row    = existing.data[0]
            total  = row["total_potholes"]  + len(detections)
            high   = row["high_count"]   + sum(1 for d in detections if d["severity"] == "High")
            medium = row["medium_count"] + sum(1 for d in detections if d["severity"] == "Medium")
            low    = row["low_count"]    + sum(1 for d in detections if d["severity"] == "Low")
            by_brgy = row.get("by_barangay") or {}
            by_src  = row.get("by_source")   or {}

            if barangay:
                by_brgy[barangay] = by_brgy.get(barangay, 0) + len(detections)
            by_src[source] = by_src.get(source, 0) + len(detections)

            supabase.table("daily_reports").update({
                "total_potholes": total,
                "high_count":     high,
                "medium_count":   medium,
                "low_count":      low,
                "by_barangay":    by_brgy,
                "by_source":      by_src,
                "generated_at":   datetime.now().isoformat(),
            }).eq("report_date", today).execute()
        else:
            by_brgy = {barangay: len(detections)} if barangay else {}
            by_src  = {source: len(detections)}
            supabase.table("daily_reports").insert({
                "report_date":    today,
                "total_potholes": len(detections),
                "high_count":     sum(1 for d in detections if d["severity"] == "High"),
                "medium_count":   sum(1 for d in detections if d["severity"] == "Medium"),
                "low_count":      sum(1 for d in detections if d["severity"] == "Low"),
                "by_barangay":    by_brgy,
                "by_source":      by_src,
            }).execute()
    except Exception as e:
        print(f"⚠️  Daily report update failed: {e}")


# ── Detection Routes ───────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "ok",
        "model":     "loaded" if model else "missing",
        "supabase":  "connected" if supabase else "disconnected",
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/pothole/detect", methods=["POST"])
def detect_image():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file         = request.files["file"]
    barangay     = request.form.get("barangay", "")
    location     = request.form.get("location", "")
    analyze_only = request.form.get("analyze_only", "false").lower() == "true"
    raw_bytes    = file.read()

    nparr = np.frombuffer(raw_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Cannot decode image"}), 400

    detections = run_yolo(frame)

    # If analyze_only=true, just return detections — do NOT save to DB or storage
    if analyze_only:
        return jsonify({
            "success":           True,
            "potholes_detected": len(detections),
            "potholes":          detections,
            "image_url":         "",
            "saved":             False,
        })

    # Otherwise upload and save
    filename  = f"{uuid.uuid4()}.jpg"
    image_url = upload_storage(raw_bytes, filename, folder="images")

    for det in detections:
        save_detection(det, "image", image_url=image_url,
                       barangay=barangay, location_label=location)

    update_daily_report(detections, barangay=barangay, source="image")

    return jsonify({
        "success":           True,
        "potholes_detected": len(detections),
        "potholes":          detections,
        "image_url":         image_url,
        "saved":             True,
    })


@app.route("/api/pothole/detect-webcam", methods=["POST"])
def detect_webcam():
    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "No frame"}), 400

    barangay = data.get("barangay", "")
    location = data.get("location", "")
    raw      = base64.b64decode(data["frame"].split(",")[-1])
    nparr    = np.frombuffer(raw, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Cannot decode frame"}), 400

    detections = run_yolo(frame)

    image_url = ""
    if detections:
        filename  = f"webcam_{uuid.uuid4()}.jpg"
        image_url = upload_storage(raw, filename, folder="webcam")
        for det in detections:
            save_detection(det, "webcam", image_url=image_url,
                           barangay=barangay, location_label=location)
        update_daily_report(detections, barangay=barangay, source="webcam")

    return jsonify({
        "success":           True,
        "potholes_detected": len(detections),
        "potholes":          detections,
        "image_url":         image_url,
    })


@app.route("/api/pothole/detect-video", methods=["POST"])
def detect_video():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file      = request.files["file"]
    barangay  = request.form.get("barangay", "")
    raw_bytes = file.read()

    ext      = os.path.splitext(file.filename or "v.mp4")[1] or ".mp4"
    filename = f"{uuid.uuid4()}{ext}"
    video_url = upload_storage(raw_bytes, filename, folder="videos")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(raw_bytes); tmp.close()

    cap      = cv2.VideoCapture(tmp.name)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25
    interval = max(1, int(fps * 2))
    all_dets = []
    idx      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            dets = run_yolo(frame)
            for d in dets:
                d["frame_number"]  = idx
                d["timestamp_sec"] = round(idx / fps, 1)
                save_detection(d, "video", video_url=video_url,
                               barangay=barangay, frame_number=idx)
            all_dets.extend(dets)
        idx += 1

    cap.release()
    os.unlink(tmp.name)
    update_daily_report(all_dets, barangay=barangay, source="video")

    sev = {"High": 0, "Medium": 0, "Low": 0}
    for d in all_dets:
        sev[d["severity"]] = sev.get(d["severity"], 0) + 1

    return jsonify({
        "success":           True,
        "potholes_detected": len(all_dets),
        "potholes":          all_dets,
        "severity_summary":  sev,
        "video_url":         video_url,
    })


# ── Stats / History ────────────────────────────────────────────

@app.route("/api/stats/dashboard", methods=["GET"])
def dashboard_stats():
    if not supabase:
        return jsonify({"success": True, "total": 0, "today": 0, "this_week": 0,
                        "critical": 0, "by_source": {}}), 200
    try:
        now   = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week  = now - timedelta(weeks=1)

        rows = supabase.table("pothole_detections") \
            .select("severity,detected_at,source").execute().data or []

        return jsonify({
            "success":   True,
            "total":     len(rows),
            "today":     sum(1 for r in rows if r["detected_at"] >= today.isoformat()),
            "this_week": sum(1 for r in rows if r["detected_at"] >= week.isoformat()),
            "critical":  sum(1 for r in rows if r["severity"] == "High"),
            "by_source": {
                "image":  sum(1 for r in rows if r.get("source") == "image"),
                "video":  sum(1 for r in rows if r.get("source") == "video"),
                "webcam": sum(1 for r in rows if r.get("source") in ["webcam", "dashcam"]),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats/history", methods=["GET"])
def get_history():
    if not supabase:
        return jsonify({"success": True, "total": 0, "daily_chart": [],
                        "by_barangay": {}, "by_source": {},
                        "severity_totals": {}, "recent": []}), 200
    period   = request.args.get("period", "1month")
    barangay = request.args.get("barangay", "")
    now      = datetime.now(timezone.utc)
    since    = {
        "1week":   now - timedelta(weeks=1),
        "1month":  now - timedelta(days=30),
        "6months": now - timedelta(days=182),
        "1year":   now - timedelta(days=365),
    }.get(period, now - timedelta(days=30))

    try:
        q = supabase.table("pothole_detections") \
            .select("id,detected_at,severity,confidence,source,barangay,image_url") \
            .gte("detected_at", since.isoformat())
        if barangay:
            q = q.eq("barangay", barangay)
        rows = q.order("detected_at", desc=True).execute().data or []

        daily, by_brgy, by_src, sev_totals = {}, {}, {}, {"High": 0, "Medium": 0, "Low": 0}
        for r in rows:
            day = r["detected_at"][:10]
            if day not in daily:
                daily[day] = {"date": day, "total": 0, "High": 0, "Medium": 0, "Low": 0}
            daily[day]["total"] += 1
            daily[day][r["severity"]] = daily[day].get(r["severity"], 0) + 1
            b = r.get("barangay") or "Unknown"
            by_brgy[b] = by_brgy.get(b, 0) + 1
            s = r.get("source", "unknown")
            by_src[s]  = by_src.get(s, 0) + 1
            sev_totals[r["severity"]] = sev_totals.get(r["severity"], 0) + 1

        return jsonify({
            "success":         True,
            "period":          period,
            "since":           since.isoformat(),
            "total":           len(rows),
            "daily_chart":     sorted(daily.values(), key=lambda x: x["date"]),
            "by_barangay":     by_brgy,
            "by_source":       by_src,
            "severity_totals": sev_totals,
            "recent":          rows[:20]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Daily Report ───────────────────────────────────────────────

@app.route("/api/reports/daily", methods=["GET"])
def get_daily_report():
    if not supabase:
        return jsonify({"error": "Supabase not connected"}), 503
    report_date = request.args.get("date", date.today().isoformat())
    try:
        day_start = f"{report_date}T00:00:00+00:00"
        day_end   = f"{report_date}T23:59:59+00:00"
        rows      = supabase.table("pothole_detections") \
            .select("*") \
            .gte("detected_at", day_start) \
            .lte("detected_at", day_end) \
            .execute().data or []

        priority = {"High": 0, "Medium": 1, "Low": 2}
        rows.sort(key=lambda r: priority.get(r.get("severity", "Low"), 3))

        sev = {"High": 0, "Medium": 0, "Low": 0}
        for r in rows:
            sev[r["severity"]] = sev.get(r["severity"], 0) + 1

        by_brgy = {}
        for r in rows:
            b = r.get("barangay") or "Unknown"
            if b not in by_brgy:
                by_brgy[b] = {"High": 0, "Medium": 0, "Low": 0, "total": 0}
            by_brgy[b][r["severity"]] += 1
            by_brgy[b]["total"] += 1

        return jsonify({
            "success":         True,
            "date":            report_date,
            "total":           len(rows),
            "severity_counts": sev,
            "by_barangay":     by_brgy,
            "detections":      rows,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reports/daily/download", methods=["GET"])
def download_daily_report():
    if not supabase:
        return jsonify({"error": "Supabase not connected"}), 503
    report_date = request.args.get("date", date.today().isoformat())
    try:
        day_start = f"{report_date}T00:00:00+00:00"
        day_end   = f"{report_date}T23:59:59+00:00"
        rows      = supabase.table("pothole_detections") \
            .select("*").gte("detected_at", day_start) \
            .lte("detected_at", day_end).execute().data or []

        priority = {"High": 0, "Medium": 1, "Low": 2}
        rows.sort(key=lambda r: priority.get(r.get("severity", "Low"), 3))

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Priority", "Detected At", "Severity", "Confidence (%)",
            "Barangay", "Location", "Source", "Image URL", "Bounding Box"
        ])
        for i, r in enumerate(rows, 1):
            writer.writerow([
                i,
                r.get("detected_at", ""),
                r.get("severity", ""),
                round(r.get("confidence", 0), 1),
                r.get("barangay", ""),
                r.get("location_label", ""),
                r.get("source", ""),
                r.get("image_url", ""),
                r.get("bbox", ""),
            ])

        csv_bytes = output.getvalue().encode("utf-8")
        return send_file(
            io.BytesIO(csv_bytes),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"pothole_daily_report_{report_date}.csv"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Barangay Alerts ────────────────────────────────────────────

@app.route("/api/barangays", methods=["GET"])
def get_barangays():
    if not supabase:
        return jsonify({"success": True, "barangays": []}), 200
    try:
        res = supabase.table("barangay_alert_config") \
            .select("*").order("barangay_name").execute()
        return jsonify({"success": True, "barangays": res.data or []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/barangays", methods=["POST"])
def upsert_barangay():
    if not supabase:
        return jsonify({"error": "Supabase not connected"}), 503
    data = request.get_json()
    if not data.get("barangay_name") or not data.get("recipient_email"):
        return jsonify({"error": "barangay_name and recipient_email required"}), 400
    try:
        res = supabase.table("barangay_alert_config").upsert({
            "barangay_name":   data["barangay_name"],
            "recipient_email": data["recipient_email"],
            "alert_threshold": int(data.get("alert_threshold", 5)),
            "is_active":       bool(data.get("is_active", True)),
            "updated_at":      datetime.now().isoformat()
        }, on_conflict="barangay_name").execute()
        return jsonify({"success": True, "data": res.data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alerts/generate", methods=["POST"])
def generate_alert():
    if not supabase:
        return jsonify({"error": "Supabase not connected"}), 503
    data     = request.get_json() or {}
    barangay = data.get("barangay", "")
    period   = data.get("period", "1week")
    do_email = data.get("send_email", True)

    now   = datetime.now(timezone.utc)
    since = {
        "1week":   now - timedelta(weeks=1),
        "1month":  now - timedelta(days=30),
        "6months": now - timedelta(days=182),
        "1year":   now - timedelta(days=365),
    }.get(period, now - timedelta(weeks=1))

    try:
        q = supabase.table("pothole_detections").select("*") \
            .gte("detected_at", since.isoformat()).order("detected_at", desc=True)
        if barangay:
            q = q.eq("barangay", barangay)
        rows = q.execute().data or []

        priority = {"High": 0, "Medium": 1, "Low": 2}
        rows.sort(key=lambda r: priority.get(r.get("severity", "Low"), 3))

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Priority", "Detected At", "Barangay", "Severity",
                          "Confidence (%)", "Source", "Image URL", "Location", "Bounding Box"])
        for i, r in enumerate(rows, 1):
            writer.writerow([i, r.get("detected_at", ""), r.get("barangay", ""),
                              r.get("severity", ""), round(r.get("confidence", 0), 1),
                              r.get("source", ""), r.get("image_url", ""),
                              r.get("location_label", ""), r.get("bbox", "")])

        csv_content  = output.getvalue()
        csv_filename = f"pothole_{barangay.replace(' ', '_')}_{period}_{now.strftime('%Y%m%d')}.csv"

        brgy_cfg = None
        if barangay:
            cfg = supabase.table("barangay_alert_config") \
                .select("*").eq("barangay_name", barangay).execute()
            if cfg.data:
                brgy_cfg = cfg.data[0]

        email_status = "skipped"
        if do_email and brgy_cfg:
            email_status = send_email_csv(
                brgy_cfg["recipient_email"], barangay, period,
                len(rows), csv_content, csv_filename
            )

        supabase.table("alert_history").insert({
            "barangay":        barangay,
            "recipient_email": brgy_cfg["recipient_email"] if brgy_cfg else "none",
            "pothole_count":   len(rows),
            "period_start":    since.isoformat(),
            "period_end":      now.isoformat(),
            "csv_filename":    csv_filename,
            "status":          "sent" if email_status == "ok" else "failed"
        }).execute()

        if barangay and brgy_cfg:
            supabase.table("barangay_alert_config") \
                .update({"last_alerted_at": now.isoformat()}) \
                .eq("barangay_name", barangay).execute()

        return jsonify({
            "success":      True,
            "barangay":     barangay,
            "records":      len(rows),
            "csv_filename": csv_filename,
            "email_status": email_status,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alerts/download-csv", methods=["POST"])
def download_csv():
    if not supabase:
        return jsonify({"error": "Supabase not connected"}), 503
    data     = request.get_json() or {}
    barangay = data.get("barangay", "")
    period   = data.get("period", "1week")
    now      = datetime.now(timezone.utc)
    since    = {
        "1week":   now - timedelta(weeks=1),
        "1month":  now - timedelta(days=30),
        "6months": now - timedelta(days=182),
        "1year":   now - timedelta(days=365),
    }.get(period, now - timedelta(weeks=1))

    try:
        q = supabase.table("pothole_detections").select("*") \
            .gte("detected_at", since.isoformat()).order("detected_at", desc=True)
        if barangay:
            q = q.eq("barangay", barangay)
        rows = q.execute().data or []
        priority = {"High": 0, "Medium": 1, "Low": 2}
        rows.sort(key=lambda r: priority.get(r.get("severity", "Low"), 3))

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Priority", "Detected At", "Barangay", "Severity",
                          "Confidence (%)", "Source", "Image URL", "Location", "Bounding Box"])
        for i, r in enumerate(rows, 1):
            writer.writerow([i, r.get("detected_at", ""), r.get("barangay", ""),
                              r.get("severity", ""), round(r.get("confidence", 0), 1),
                              r.get("source", ""), r.get("image_url", ""),
                              r.get("location_label", ""), r.get("bbox", "")])

        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8")),
            mimetype="text/csv", as_attachment=True,
            download_name=f"pothole_{barangay.replace(' ', '_')}_{period}_{now.strftime('%Y%m%d')}.csv"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alerts/history", methods=["GET"])
def alert_history():
    if not supabase:
        return jsonify({"success": True, "history": []}), 200
    try:
        res = supabase.table("alert_history").select("*") \
            .order("sent_at", desc=True).limit(50).execute()
        return jsonify({"success": True, "history": res.data or []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/storage/status", methods=["GET"])
def storage_status():
    if not supabase:
        return jsonify({"error": "Supabase not connected"}), 503
    try:
        rows = supabase.table("pothole_detections") \
            .select("source,image_url,video_url").execute().data or []
        return jsonify({
            "success":       True,
            "bucket":        BUCKET,
            "supabase_url":  SUPABASE_URL,
            "total_records": len(rows),
            "files_in_storage": {
                "images": sum(1 for r in rows if r.get("image_url")),
                "videos": sum(1 for r in rows if r.get("video_url")),
            },
            "detections_by_source": {
                "image":  sum(1 for r in rows if r.get("source") == "image"),
                "video":  sum(1 for r in rows if r.get("source") == "video"),
                "webcam": sum(1 for r in rows if r.get("source") in ["webcam", "dashcam"]),
            },
            "storage_paths": {
                "images": f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/images/",
                "videos": f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/videos/",
                "webcam": f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/webcam/",
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Email ──────────────────────────────────────────────────────

def send_email_csv(recipient, barangay, period, count, csv_content, csv_filename) -> str:
    """
    Send email with CSV attachment.
    Supports Gmail App Password, Brevo, Outlook, and any SMTP provider.
    Configure via .env — see EMAIL SETUP section in README.
    """
    smtp_user = os.getenv("SMTP_EMAIL", "")
    smtp_pass = os.getenv("SMTP_PASSWORD", "")
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_ssl  = os.getenv("SMTP_SSL", "true").lower() == "true"

    if not smtp_user or not smtp_pass:
        print("⚠️  No SMTP credentials — email skipped. CSV can still be downloaded.")
        return "no_credentials"

    try:
        msg           = MIMEMultipart()
        msg["From"]   = smtp_user
        msg["To"]     = recipient
        msg["Subject"] = f"Pothole Alert — {barangay} ({period})"
        body = (
            f"Barangay: {barangay}\n"
            f"Period: {period}\n"
            f"Detections: {count}\n"
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n"
            f"See attached CSV report sorted by priority (High → Medium → Low).\n\n"
            f"— SOFTDESBG6 Pothole Detection System"
        )
        msg.attach(MIMEText(body, "plain"))
        att = MIMEBase("application", "octet-stream")
        att.set_payload(csv_content.encode("utf-8"))
        encoders.encode_base64(att)
        att.add_header("Content-Disposition", f"attachment; filename={csv_filename}")
        msg.attach(att)

        if smtp_ssl:
            # SSL connection (Gmail default port 465)
            with smtplib.SMTP_SSL(smtp_host, smtp_port) as s:
                s.login(smtp_user, smtp_pass)
                s.sendmail(smtp_user, recipient, msg.as_string())
        else:
            # STARTTLS connection (Brevo/Outlook port 587)
            with smtplib.SMTP(smtp_host, smtp_port) as s:
                s.ehlo()
                s.starttls()
                s.login(smtp_user, smtp_pass)
                s.sendmail(smtp_user, recipient, msg.as_string())

        print(f"✅ Email sent to {recipient}")
        return "ok"
    except Exception as e:
        print(f"⚠️  Email failed: {e}")
        return f"error: {e}"


# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 SOFTDESBG6 — Pothole Detection API")
    print(f"   YOLO  : {'loaded' if model else 'MISSING — place best.pt in backend/weights/'}")
    print(f"   DB    : {'connected' if supabase else 'disconnected'}")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)