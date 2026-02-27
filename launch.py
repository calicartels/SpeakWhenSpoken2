import argparse
import os
import sys
import time
import webbrowser

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DASHBOARD_PORT = 8888
VLLM_PORT = 8000


def env(name, default=""):
    return os.environ.get(name, default).strip()


def ssh_cmd(ip, port, remote_cmd):
    cmd = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p {port} root@{ip} '{remote_cmd}'"
    return os.system(cmd)


def wait_vllm(ip, port, timeout=120):
    """Poll vLLM health endpoint via SSH until it responds."""
    print(f"Waiting for vLLM to be ready (up to {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        r = ssh_cmd(ip, port, f"curl -sf http://localhost:{VLLM_PORT}/health >/dev/null 2>&1")
        if r == 0:
            print("vLLM is ready.")
            return True
        time.sleep(5)
    print("vLLM did not become healthy in time. Check /tmp/vllm.log on the instance.")
    return False


def main():
    ap = argparse.ArgumentParser(description="Launch dashboard: vLLM + server, tunnel, http")
    ap.add_argument("--no-server", action="store_true", help="Skip starting server/vLLM on Vast")
    ap.add_argument("--no-vllm", action="store_true", help="Skip starting vLLM (already running)")
    args = ap.parse_args()

    ip = env("VAST_IP")
    port = env("VAST_SSH_PORT")
    remote_path = env("VAST_REMOTE_PATH") or "/workspace/SpeakWhenSpoken2"

    if not ip or not port:
        print("Set VAST_IP and VAST_SSH_PORT in .env (copy from .env.example)")
        sys.exit(1)

    if not args.no_server:
        print("Stopping old processes...")
        ssh_cmd(ip, port, "pkill -f '[v]llm serve' || true; pkill -f '[s]erver.py' || true")
        time.sleep(2)

        if not args.no_vllm:
            print("Starting vLLM (Voxtral Realtime)...")
            ssh_cmd(ip, port, f"cd {remote_path} && nohup bash start_vllm.sh > /tmp/vllm.log 2>&1 &")
            if not wait_vllm(ip, port):
                sys.exit(1)

        print("Starting server.py...")
        ssh_cmd(ip, port, f"cd {remote_path} && nohup python3 server.py > /tmp/speakwhen.log 2>&1 &")
        print("Waiting for models to load (~15s)...")
        time.sleep(15)

    print("Starting SSH tunnel (8765)...")
    r = os.system(f"ssh -f -N -o StrictHostKeyChecking=no -o ConnectTimeout=15 -L 8765:localhost:8765 -p {port} root@{ip}")
    if r != 0:
        print("Tunnel failed. Is the instance running?")
        sys.exit(1)

    time.sleep(2)

    url = f"http://127.0.0.1:{DASHBOARD_PORT}/dashboard.html"
    print("Opening dashboard at", url)
    webbrowser.open(url)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    import http.server
    import socketserver
    print(f"HTTP server on port {DASHBOARD_PORT} (Ctrl+C to stop)")
    with socketserver.TCPServer(("", DASHBOARD_PORT), http.server.SimpleHTTPRequestHandler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
