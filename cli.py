import sys

# Ensure UTF-8 output on Windows standard console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stdin, 'reconfigure'):
    sys.stdin.reconfigure(encoding='utf-8')

from langchain_core.messages import HumanMessage
from .multiagent.graph import multiagent_app

def main():
    print("=" * 65)
    print(" 🤖 MULTI-AGENT WORKFLOW CLI (Based on idea.txt)")
    print(" Flow: Query Router -> (RAG | Order Tool | Support) -> Validator")
    print(" Type 'exit' or 'quit' to stop.")
    print("=" * 65 + "\n")

    thread_id = "cli_session_1"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\n👤 Customer: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting CLI session. Goodbye!")
                break
            if not user_input:
                continue

            print("\n🔄 Executing Graph Pipeline...")
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "draft_response": None,
                "validated_response": None,
                "validation_status": None,
                "validator_feedback": None,
                "intent": None,
                "customer_id": "CUST101",
                "agent_logs": []
            }

            final_state = multiagent_app.invoke(initial_state, config=config)
            reply = final_state.get("validated_response") or final_state.get("draft_response")
            print(f"\n🤖 Assistant:\n{reply}\n")
            print("-" * 55)

        except KeyboardInterrupt:
            print("\nSession stopped.")
            break
        except Exception as e:
            print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()
