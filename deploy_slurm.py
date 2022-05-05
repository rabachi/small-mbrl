import argparse
from deploy import deploy_losses
from main2 import experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Slurm deployment')
    parser.add_argument('--deploy_num', type=int, default=0, help='The deployment number')
    args = parser.parse_args()

    deploy_argss = deploy_losses()
    assert args.deploy_num < len(deploy_argss), f"Invalid deployment number: {args.deploy_num}"

    deploy_args = deploy_argss[args.deploy_num]

    print(f"Launching {args.deploy_num}, {deploy_args}")

    experiment(deploy_args)

    print(f"Finished {args.deploy_num}, {deploy_args}")
