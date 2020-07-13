import torch


def save_net(net, path):
    torch.save(
        net.state_dict(), path,
    )


def load_net(net, path):
    net.load_state_dict(torch.load(path))
