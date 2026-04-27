import os

MEMORY_FILE = "schema_memory.json"
LOG_FILE = "events.log"

def reset():
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
        print("Deleted schema_memory.json")

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        print("Deleted events.log")

    print("System reset complete.")

if __name__ == "__main__":
    reset()