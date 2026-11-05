import os
"""
=============================================================================
FUSASHIELD REST API SERVER
Backend API Service untuk Dashboard Website FusaShield
=============================================================================
Menyediakan endpoint RESTful JSON dengan dukungan CORS lengkap agar website
(React, Vue, Next.js, HTML/JS, Flutter, Laravel, PHP, dsb.) dapat mengambil data:
  - Telemetri Sensor IoT (Suhu, Kelembapan Tanah, RH Udara DHT22)
  - Hasil Klasifikasi Penyakit Daun CNN MobileNetV2
  - Keputusan Dosis & Durasi Pompa Fuzzy Mamdani/Sugeno
  - Rekomendasi Agronomi XAI dari Groq Cloud LLM
  - Riwayat Log Mitigasi (History Logs)
  - Kontrol Manual Aktuator Pompa (Trigger Command)

Zero External Dependencies: Murni menggunakan Python Standard Library!
=============================================================================
"""

import json
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# Konfigurasi Server API
HOST = "0.0.0.0"       # Mendengarkan di semua interface jaringan
PORT = int(os.environ.get("PORT", 5000))            # Port default API: http://localhost:5000

# Muat variabel environment dari .env jika ada
def _muat_env(path=".env"):
    """Muat variabel environment dari file .env lokal secara otomatis."""
    lokasi = os.path.join(os.path.dirname(os.path.abspath(__file__)), path) if "__file__" in globals() else path
    if os.path.exists(lokasi):
        try:
            with open(lokasi, "r", encoding="utf-8") as f:
                for line in f:
                    b = line.strip()
                    if b and not b.startswith("#") and "=" in b:
                        k, v = b.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception as e:
            print(f"[WARN] Gagal membaca {path}: {e}")

_muat_env()

# Konfigurasi Database Firebase & Supabase (Dari .env)
FIREBASE_DB_URL = os.environ.get("FIREBASE_DB_URL", "")
FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY", "")
NODE_PATH = os.environ.get("NODE_PATH", "/monitoring/cabai_01")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "fusashield-image")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Server HTTP multi-threaded untuk melayani banyak request website secara simultan."""
    daemon_threads = True


class FusaShieldAPIHandler(BaseHTTPRequestHandler):

    def _set_cors_headers(self, status_code=200, content_type="application/json"):
        """Mengatur header CORS agar website dari domain/port mana pun dapat mengakses tanpa terblokir browser."""
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self):
        """Menangani preflight request CORS dari browser frontend (React/Vue/Angular)."""
        self._set_cors_headers(204)

    def _send_json(self, data, status_code=200):
        """Mengirimkan payload respon JSON ke client website."""
        self._set_cors_headers(status_code)
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.wfile.write(body)

    def _fetch_firebase(self, path):
        """Mengambil data dari Firebase Realtime Database via REST API murni standard library."""
        url = f"{FIREBASE_DB_URL}{path}.json?auth={FIREBASE_API_KEY}"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FusaShield-WebAPI/1.0"}
            )
            with urllib.request.urlopen(req, timeout=6) as response:
                raw_bytes = response.read()
                if not raw_bytes:
                    return {}
                return json.loads(raw_bytes.decode("utf-8")) or {}
        except Exception as e:
            print(f"[ERROR Firebase] Gagal membaca {path}: {e}")
            return None

    def _put_firebase(self, path, payload):
        """Mengirimkan perintah ke Firebase via PUT (Standard Library)."""
        url = f"{FIREBASE_DB_URL}{path}.json?auth={FIREBASE_API_KEY}"
        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "FusaShield-WebAPI/1.0"
                },
                method="PUT"
            )
            with urllib.request.urlopen(req, timeout=6) as response:
                return response.status in (200, 204)
        except Exception as e:
            print(f"[ERROR Firebase] Gagal put ke {path}: {e}")
            return False

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)

        # -------------------------------------------------------------
        # 1. Root / Dokumentasi Endpoint API
        # -------------------------------------------------------------
        if path == "" or path == "/":
            doc = {
                "system": "FusaShield Precision Mitigation REST API",
                "version": "1.0.0",
                "status": "online",
                "endpoints": {
                    "GET /api/monitoring": "Seluruh data lengkap (Sensor IoT + CNN + Fuzzy + HST + Rekomendasi LLM)",
                    "GET /api/iot": "Data telemetri sensor IoT real-time (suhu, kelembapan tanah, RH udara)",
                    "GET /api/fuzzy": "Hasil inferensi Fuzzy Logic (dosis ml, durasi pompa, tingkat keparahan, status)",
                    "GET /api/cnn": "Hasil klasifikasi gejala visual daun CNN MobileNetV2",
                    "GET /api/llm": "Narasi rekomendasi agronomi XAI terkini",
                    "GET /api/status": "Status kesiapan sistem (apakah IoT lengkap & foto <= 24 jam)",
                    "GET /api/history?limit=20": "Riwayat riwayat log aksi mitigasi dan penyemprotan",
                    "POST /api/pump/trigger": "Kirim perintah manual dari website untuk menyalakan pompa"
                }
            }
            return self._send_json(doc)

        # -------------------------------------------------------------
        # 2. Endpoint: /api/monitoring (Full Telemetry + Decision)
        # Digunakan sebagai endpoint utama halaman dashboard website
        # -------------------------------------------------------------
        if path == "/api/monitoring":
            iot_data = self._fetch_firebase(NODE_PATH)
            fuzzy_data = self._fetch_firebase(f"{NODE_PATH}/fuzzy_output")

            if iot_data is None and fuzzy_data is None:
                return self._send_json({"error": "Gagal terhubung ke database cloud Firebase"}, 503)

            # Ekstrak data gabungan komprehensif
            response = {
                "success": True,
                "timestamp": int(time.time()),
                "plant_id": "cabai_01",
                "iot_sensors": {
                    "temperature": iot_data.get("temperature") if iot_data else None,
                    "soil_moisture": iot_data.get("soil_moisture") if iot_data else None,
                    "humidity_air": iot_data.get("humidity") if iot_data else None,
                    "hst": iot_data.get("hst", 41) if iot_data else 41,
                    "uptime_ms": iot_data.get("uptime_ms") if iot_data else None,
                    "soil_valid": iot_data.get("soil_valid", True) if iot_data else True
                },
                "system_readiness": {
                    "is_ready": (iot_data is not None and all(iot_data.get(k) is not None for k in ["temperature", "humidity", "soil_moisture"])) and (fuzzy_data is not None and fuzzy_data.get("foto_umur_jam") is not None and fuzzy_data.get("foto_umur_jam") <= 24.0),
                    "iot_complete": iot_data is not None and all(iot_data.get(k) is not None for k in ["temperature", "humidity", "soil_moisture"]),
                    "image_valid_24h": fuzzy_data is not None and fuzzy_data.get("foto_umur_jam") is not None and fuzzy_data.get("foto_umur_jam") <= 24.0,
                    "mode": "100% DATA RIIL (TANPA DEMO)"
                },
                "cnn_vision": {
                    "label": fuzzy_data.get("cnn_label") if fuzzy_data else "Unknown",
                    "confidence_pct": fuzzy_data.get("cnn_confidence") if fuzzy_data else 0.0,
                    "symptom_score_pct": fuzzy_data.get("cnn_symptom_score") if fuzzy_data else 0.0,
                    "image_source": fuzzy_data.get("foto_sumber") if fuzzy_data else None,
                    "image_age_hours": fuzzy_data.get("foto_umur_jam") if fuzzy_data else None,
                    "image_valid_24h": fuzzy_data.get("foto_umur_jam") is not None and fuzzy_data.get("foto_umur_jam") <= 24.0 if fuzzy_data else False,
                    "image_public_url": f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{fuzzy_data.get('foto_sumber')}" if (fuzzy_data and fuzzy_data.get("foto_sumber") and SUPABASE_URL) else None
                },
                "fuzzy_decision": {
                    "dosis_ml": fuzzy_data.get("dosis_rekomendasi_ml", 0.0) if fuzzy_data else 0.0,
                    "durasi_detik": fuzzy_data.get("durasi_pompa_detik", 0.0) if fuzzy_data else 0.0,
                    "pump_command": fuzzy_data.get("pump_command", "OFF") if fuzzy_data else "OFF",
                    "status_mitigasi": fuzzy_data.get("status_mitigasi", "STANDBY") if fuzzy_data else "STANDBY",
                    "fase_hst": fuzzy_data.get("fase_hst") if fuzzy_data else None,
                    "tingkat_dosis": fuzzy_data.get("tingkat_dosis", "AMAN") if fuzzy_data else "AMAN",
                    "cooldown_jam": fuzzy_data.get("cooldown_jam", 12.0) if fuzzy_data else 12.0,
                    "sisa_cooldown_detik": fuzzy_data.get("sisa_cooldown_detik", 0) if fuzzy_data else 0
                },
                "llm_advisory": {
                    "text": fuzzy_data.get("rekomendasi_llm", "") if fuzzy_data else "",
                    "source": fuzzy_data.get("rekomendasi_llm_sumber", "template") if fuzzy_data else "template"
                },
                "last_updated": fuzzy_data.get("last_updated") if fuzzy_data else None
            }
            return self._send_json(response)

        # -------------------------------------------------------------
        # 3. Endpoint: /api/iot (Hanya Sensor Lapangan)
        # -------------------------------------------------------------
        if path == "/api/iot":
            iot_data = self._fetch_firebase(NODE_PATH)
            if iot_data is None:
                return self._send_json({"error": "Gagal membaca sensor IoT"}, 503)
            return self._send_json({
                "success": True,
                "plant_id": "cabai_01",
                "temperature": iot_data.get("temperature"),
                "soil_moisture": iot_data.get("soil_moisture"),
                "humidity": iot_data.get("humidity"),
                "hst": iot_data.get("hst", 41),
                "soil_valid": iot_data.get("soil_valid", True),
                "uptime_ms": iot_data.get("uptime_ms"),
                "ack": iot_data.get("ack")
            })

        # -------------------------------------------------------------
        # 4. Endpoint: /api/fuzzy (Hanya Keputusan Fuzzy Logic)
        # -------------------------------------------------------------
        if path == "/api/fuzzy":
            fuzzy_data = self._fetch_firebase(f"{NODE_PATH}/fuzzy_output")
            if fuzzy_data is None:
                return self._send_json({"error": "Gagal membaca output fuzzy"}, 503)
            return self._send_json({
                "success": True,
                "plant_id": "cabai_01",
                "dosis_rekomendasi_ml": fuzzy_data.get("dosis_rekomendasi_ml", 0.0),
                "durasi_pompa_detik": fuzzy_data.get("durasi_pompa_detik", 0.0),
                "pump_command": fuzzy_data.get("pump_command", "OFF"),
                "status_mitigasi": fuzzy_data.get("status_mitigasi"),
                "fase_hst": fuzzy_data.get("fase_hst"),
                "tingkat_dosis": fuzzy_data.get("tingkat_dosis"),
                "cooldown_jam": fuzzy_data.get("cooldown_jam"),
                "sisa_cooldown_detik": fuzzy_data.get("sisa_cooldown_detik"),
                "last_updated": fuzzy_data.get("last_updated")
            })

        # -------------------------------------------------------------
        # 5. Endpoint: /api/cnn (Hanya Hasil Deteksi Daun)
        # -------------------------------------------------------------
        if path == "/api/cnn":
            fuzzy_data = self._fetch_firebase(f"{NODE_PATH}/fuzzy_output")
            if fuzzy_data is None:
                return self._send_json({"error": "Gagal membaca data CNN"}, 503)
            foto_src = fuzzy_data.get("foto_sumber")
            umur_jam = fuzzy_data.get("foto_umur_jam")
            valid_24h = (umur_jam is not None and umur_jam <= 24.0)
            pub_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{foto_src}" if (foto_src and SUPABASE_URL) else None
            return self._send_json({
                "success": True,
                "plant_id": "cabai_01",
                "label": fuzzy_data.get("cnn_label"),
                "confidence_pct": fuzzy_data.get("cnn_confidence"),
                "symptom_score_pct": fuzzy_data.get("cnn_symptom_score"),
                "foto_sumber": foto_src,
                "foto_public_url": pub_url,
                "foto_umur_jam": umur_jam,
                "foto_valid_24h": valid_24h
            })

        # -------------------------------------------------------------
        # 6. Endpoint: /api/llm (Hanya Saran Agronomi LLM)
        # -------------------------------------------------------------
        if path == "/api/llm":
            fuzzy_data = self._fetch_firebase(f"{NODE_PATH}/fuzzy_output")
            if fuzzy_data is None:
                return self._send_json({"error": "Gagal membaca data LLM"}, 503)
            return self._send_json({
                "success": True,
                "plant_id": "cabai_01",
                "advisory": fuzzy_data.get("rekomendasi_llm", ""),
                "source": fuzzy_data.get("rekomendasi_llm_sumber", "template"),
                "last_updated": fuzzy_data.get("last_updated")
            })

        # -------------------------------------------------------------
        # 7. Endpoint: /api/history (Riwayat Log Penyemprotan & Analisis)
        # -------------------------------------------------------------
        if path == "/api/history":
            limit = 20
            if "limit" in query:
                try:
                    limit = int(query["limit"][0])
                except ValueError:
                    pass

            history_raw = self._fetch_firebase("/history_log/cabai_01")
            logs = []
            if isinstance(history_raw, dict):
                for key, val in history_raw.items():
                    if isinstance(val, dict):
                        item = dict(val)
                        item["log_id"] = key
                        logs.append(item)
            elif isinstance(history_raw, list):
                logs = [x for x in history_raw if x]

            # Urutkan dari data paling terkini
            logs.reverse()
            logs = logs[:limit]

            return self._send_json({
                "success": True,
                "total_fetched": len(logs),
                "limit": limit,
                "logs": logs
            })

        # Endpoint tidak ditemukan
        self._send_json({"error": "Endpoint tidak ditemukan", "path": path}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        # -------------------------------------------------------------
        # Endpoint: POST /api/pump/trigger
        # Menyalakan pompa manual dari tombol di website
        # Body JSON opsional: {"dosis_ml": 25.0, "durasi_detik": 5.0}
        # -------------------------------------------------------------
        if path == "/api/pump/trigger":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                payload_in = json.loads(body.decode("utf-8")) if body else {}
            except json.JSONDecodeError:
                payload_in = {}

            dosis = float(payload_in.get("dosis_ml", 25.0))
            durasi = float(payload_in.get("durasi_detik", dosis / 5.0))
            sekarang = int(time.time())

            perintah = {
                "id": sekarang,
                "dosis_ml": dosis,
                "durasi_detik": durasi,
                "issued_at": sekarang,
                "status_mitigasi": "MANUAL_TRIGGER_VIA_WEBSITE"
            }

            url_cmd = f"{NODE_PATH}/command"
            sukses = self._put_firebase(url_cmd, perintah)

            if sukses:
                return self._send_json({
                    "success": True,
                    "message": "Perintah pompa berhasil dikirim ke ESP32",
                    "command": perintah
                }, 200)
            else:
                return self._send_json({
                    "success": False,
                    "error": "Gagal mengirim perintah ke Firebase"
                }, 500)

        self._send_json({"error": "Endpoint POST tidak ditemukan", "path": path}, 404)


def _loop_cloud_controller():
    print("[BACKGROUND WORKER] Menjalankan pemindaian otomatis FusaShield di latar belakang...")
    try:
        import fusashield_main
        while True:
            try:
                fusashield_main.jalankan_pipeline_cloud()
                time.sleep(10)
            except Exception as err:
                print(f"[WORKER ERROR] {err}. Mengulang dalam 10 detik...")
                time.sleep(10)
    except Exception as e:
        print(f"[WORKER INIT ERROR] Gagal mengimpor fusashield_main: {e}")


def run_api_server(host=HOST, port=PORT):
    import threading
    worker_thread = threading.Thread(target=_loop_cloud_controller, daemon=True)
    worker_thread.start()

    server_address = (host, port)
    httpd = ThreadedHTTPServer(server_address, FusaShieldAPIHandler)
    print("=" * 65)
    print(" [FUSASHIELD REST API SERVER BERJALAN]")
    print(f" URL API Lokal: http://localhost:{port}")
    print(f" URL Jaringan : http://127.0.0.1:{port}")
    print(" Status CORS  : AKTIF (Aman di-fetch dari React/Vue/HTML)")
    print(" Dokumentasi  : Buka http://localhost:5000 di browser")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Server API dihentikan.")
        httpd.server_close()


if __name__ == "__main__":
    run_api_server()
