import sys
import os
import argparse

# Add current directory to path so that 'src' is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Recipe Recommender System")
    parser.add_argument("--mode", type=str, choices=["train", "serve", "all"], default="all",
                        help="Mode: train, serve, or all (train and then serve)")
    
    args = parser.parse_args()

    if args.mode in ["train", "all"]:
        print("Starting Training and Evaluation Phase...")
        from src.train_and_evaluate_all import main as train_main
        train_main()

    if args.mode in ["serve", "all"]:
        print("Starting FastAPI Server on port 2222...")
        import uvicorn
        from src.api import app
        uvicorn.run(app, host="0.0.0.0", port=2222)

if __name__ == "__main__":
    main()
