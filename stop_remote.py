
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def main():
    ip = os.environ.get("VAST_IP", "").strip()
    port = os.environ.get("VAST_SSH_PORT", "").strip()
    if not ip or not port:
        print("Set VAST_IP and VAST_SSH_PORT in .env")
        sys.exit(1)

    print(f"Connecting to {ip}:{port}...")
    cmd = f'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p {port} root@{ip} "pkill -f \'[v]llm serve\' || true; pkill -f \'[s]erver.py\' || true; echo ok"'
    r = os.system(cmd)
    exit_code = os.waitstatus_to_exitcode(r) if hasattr(os, 'waitstatus_to_exitcode') else r >> 8
    if exit_code == 0:
        print("vLLM + server stopped (or were not running)")
    else:
        print("SSH connection failed")

if __name__ == "__main__":
    main()
