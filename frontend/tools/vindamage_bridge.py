"""Local-only, optional bridge: Python backend state -> React warning.html.

Usage: python tools/vindamage_bridge.py [--host 127.0.0.1] [--port 8765]
POST /state with a JSON state in docs/VINDAMAGE_FE.md, GET /state returns last state.
Bind to loopback only: no auth is provided. No default synthetic detections.
"""
from __future__ import annotations
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

MAX_BYTES = 12 * 1024 * 1024
STATE: dict = {}
LAST_UPDATE = [0.0]
LOCK = Lock()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, data: dict):
        payload = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path != '/state':
            return self._send(404, {'error': 'Not found'})
        with LOCK:
            state = STATE.copy()
            age = time.monotonic() - LAST_UPDATE[0]
        if not state or age > 2.5:
            return self._send(503, {'error': 'Waiting for fresh backend state (<2.5s)'})
        self._send(200, state)

    def do_POST(self):
        if self.path != '/state':
            return self._send(404, {'error': 'Not found'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > MAX_BYTES:
                return self._send(413, {'error': 'Body must be 1 to 12MB'})
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or not isinstance(body.get('tracks'), list) or not isinstance(body.get('dets'), list):
                return self._send(422, {'error': 'Expected BE state dict with tracks and dets arrays'})
            with LOCK:
                STATE.clear()
                STATE.update(body)
                LAST_UPDATE[0] = time.monotonic()
            return self._send(200, {'ok': True})
        except (ValueError, UnicodeError) as e:
            return self._send(400, {'error': str(e)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if args.host not in ('127.0.0.1', 'localhost', '::1'):
        parser.error('The bridge accepts only loopback hosts because it has no authentication.')
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'VinDamage bridge at http://{args.host}:{args.port}/state (Ctrl+C to stop)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == '__main__':
    main()
