import os
import argparse
from rag_chain import answer_question

def main():
    print("=== Bengali Physics Tutor (HSC) ===")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            user_input = input("\nAsk a Physics question (in Bengali or English): ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue
                
            response = answer_question(user_input)
            print("-" * 50)
            print("Tutor:", response)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
