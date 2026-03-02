import os
import httpx

from dotenv import load_dotenv
load_dotenv()


def clear_all():
    api_key = os.environ.get("SUPERMEMORY_API_KEY")
    if not api_key:
        print("SUPERMEMORY_API_KEY not set")
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    with httpx.Client(base_url="https://api.supermemory.ai/api/v1") as c:
        res = c.get("/memories", headers=headers, params={"limit": 1000})
        res.raise_for_status()
        data = res.json()

        memories = data.get("memories", data.get("results", []))
        if not memories:
            print("No memories found")
            return

        print(f"Found {len(memories)} memories, deleting...")
        deleted = 0
        for mem in memories:
            mid = mem.get("id")
            if mid:
                r = c.delete(f"/memories/{mid}", headers=headers)
                if r.status_code == 200:
                    deleted += 1
        print(f"Deleted {deleted} memories")


if __name__ == "__main__":
    confirm = input("Wipe all Supermemory data? (yes/no): ")
    if confirm.lower() == "yes":
        clear_all()
    else:
        print("Cancelled")
