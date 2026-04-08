# Define 3 example tasks with expected answers for grading
def get_task(task_name: str):
    tasks = {
        "easy": {
            "email": {
                "sender": "alice@example.com",
                "subject": "Meeting Request",
                "body": "Can we have a short sync tomorrow?"
            },
            "expected_priority": "low",
            "expected_action": "acknowledge",
            "expected_reply": ""
        },
        "medium": {
            "email": {
                "sender": "bob@example.com",
                "subject": "Server Down",
                "body": "The production server is down, urgent help needed!"
            },
            "expected_priority": "high",
            "expected_action": "escalate",
            "expected_reply": ""
        },
        "hard": {
            "email": {
                "sender": "ceo@example.com",
                "subject": "Project Update",
                "body": "Please provide a detailed status update on Project X."
            },
            "expected_priority": "high",
            "expected_action": "reply",
            "expected_reply": "Project X is on track and we are meeting deadlines."
        }
    }
    return tasks.get(task_name, tasks["easy"])