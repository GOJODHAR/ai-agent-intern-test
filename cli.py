import sys
import argparse
from agent import AsterRowAgent
from conversation import ConversationManager
from response import AgentResponse

def main():
    parser = argparse.ArgumentParser(description="Aster & Row AI Support Agent CLI")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug trace logging")
    args = parser.parse_args()

    # Load agent
    print("Initializing support agent database...")
    agent = AsterRowAgent()
    conv_mgr = ConversationManager(session_id="cli-session")
    
    print("\nAster & Row Support Agent is Ready.")
    print("Type 'exit', 'quit', or 'clear' to manage the session.")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            if user_input.lower() == 'clear':
                conv_mgr.clear()
                print("Session cleared.")
                continue

            # Run agent loop
            response, trace = agent.chat("cli-session", user_input, conv_mgr)
            
            # Print agent response
            print(f"\nAgent: {response.answer}")
            
            # Print citations if available
            if response.sources:
                print("\nSources:")
                for source in response.sources:
                    print(f"  - {source.filename} → {source.heading}")
            
            # Print handoff indicator
            if response.handoff:
                print(f"\n[⚠️ Human assistance recommended: {response.handoff_reason}]")
                
            # If debug mode, print execution trace
            if args.debug:
                print("\n" + trace.format_trace())
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()
