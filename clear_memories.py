import os
import httpx
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("clear_memories")

def clear_all_memories():
    api_key = os.environ.get("SUPERMEMORY_API_KEY")
    if not api_key:
        log.error("SUPERMEMORY_API_KEY environment variable is missing.")
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    log.info("Fetching existing memories from Supermemory...")
    try:
        # We must use standard HTTPX because the SDK doesn't natively support bulk delete effectively
        with httpx.Client(base_url="https://api.supermemory.ai/api/v1") as client:
            # 1. Fetch memories
            res = client.get("/memories", headers=headers, params={"limit": 1000})
            res.raise_for_status()
            data = res.json()
            
            memories = data.get("memories", data.get("results", []))
            
            if not memories:
                log.info("No memories found. Database is already clean.")
                return
            
            log.info(f"Found {len(memories)} memories. Proceeding to delete all...")
            
            # 2. Delete each memory
            deleted_count = 0
            for mem in memories:
                mem_id = mem.get("id")
                if mem_id:
                    del_res = client.delete(f"/memories/{mem_id}", headers=headers)
                    if del_res.status_code == 200:
                        deleted_count += 1
                        log.info(f"Deleted memory > {mem_id}")
                    else:
                        log.warning(f"Failed to delete {mem_id}: {del_res.status_code} {del_res.text}")
                        
            log.info(f"Successfully deleted {deleted_count} memories from Supermemory.")
            
    except Exception as e:
        log.error(f"Error communicating with Supermemory API: {e}")

if __name__ == "__main__":
    confirm = input("Are you sure you want to completely WIPE your Supermemory graph? (yes/no): ")
    if confirm.lower() == 'yes':
        clear_all_memories()
    else:
        print("Cancelled.")
