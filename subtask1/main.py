import argparse
from rl_optimization.train import train
from rl_optimization.test import test
from scripts.run_simulation import run_simulation

def main():
    parser = argparse.ArgumentParser(description='LEO to Vehicle Network Optimization')
    parser.add_argument('--mode', choices=['train', 'test', 'simulate'], required=True, help='Mode to run the script in')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    elif args.mode == 'simulate':
        run_simulation()

if __name__ == "__main__":
    main()
