import yaml
from agent import AgentController


if __name__ == "__main__":
    with open ("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    class CustomArgs:
        def __init__(self, dataset, config):
            for name, value in config[dataset].items():
                setattr(self, name, value)

    # experiment = "Split5_CIFAR10"
    experiment = "Split5_MNIST"
    args = CustomArgs(experiment, config)
    # print(args.__dict__)

    admin = AgentController(experiment, args)
    # admin.run("EWC")
    approaches = ["SCP", "SI", "RWalk", "Neuron"]
    admin.run_multi(approaches)