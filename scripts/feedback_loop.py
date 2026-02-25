import argparse
import json
import time
from pathlib import Path

def submit_feedback():
    parser = argparse.ArgumentParser(description="GAAP Feedback Tool")
    parser.add_argument("--task-id", help="ID of the task to complain about", default="last")
    parser.add_argument("--rating", type=int, choices=[1, 2, 3, 4, 5], help="1=Stupid, 5=Genius")
    parser.add_argument("--comment", type=str, help="What went wrong?")
    
    args = parser.parse_args()
    
    # In production, fetch last task ID from local state
    task_id = args.task_id if args.task_id != "last" else f"task_{int(time.time())}"
    
    feedback = {
        "task_id": task_id,
        "rating": args.rating,
        "comment": args.comment,
        "timestamp": time.time(),
        "type": "USER_FEEDBACK"
    }
    
    # Store in "Negative Memory" folder
    feedback_dir = Path("gaap/.kilocode/memory/feedback/")
    feedback_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = feedback_dir / f"feedback_{task_id}.json"
    with open(file_path, "w") as f:
        json.dump(feedback, f, indent=2)
        
    print(f"✅ Feedback received for {task_id}.")
    print("🧠 GAAP will analyze this during the next 'Dream Cycle' to prevent recurrence.")

if __name__ == "__main__":
    submit_feedback()
