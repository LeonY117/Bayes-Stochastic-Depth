import torch.optim as optim
from typing import Optional
from training.loss import CELoss, FocalLoss

from typing import List

__all__ = ["parse_scheduler", "parse_loss", "parse_optimizer", "get_dataset_classes"]


def get_dataset_classes(dataset: str) -> List[str]:
    dataset = dataset.lower()
    if dataset == "cifar10":
        classes = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    return classes


def parse_optimizer(
    model, optimizer_name: str, lr: float, weight_decay: float, **kwargs
):
    """
    Parse the optimizer name and return the optimizer
    """
    if optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=kwargs.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), weight_decay=weight_decay, lr=lr)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")
    return optimizer


def parse_scheduler(
    optimizer, scheduler_name: str, total_epochs=Optional[float], **kwargs
):
    """
    Parse the scheduler name and return the scheduler
    """
    if scheduler_name == "poly":
        # total_iters = self.total_epochs * math.ceil(len(self.dataloaders['train'].dataset)/self.batch_size)
        total_iters = total_epochs
        scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=total_iters, power=kwargs.get("power", 0.9)
        )
    elif scheduler_name == "exp":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=kwargs.get("gamma", 0.9)
        )
    elif scheduler_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get("milestones", [150, 225]),
            gamma=kwargs.get("gamma", 0.1),
        )
    else:
        scheduler = optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=total_epochs, factor=1
        )
    return scheduler


def parse_loss(loss_name: str):
    """
    Parse the loss function name and return the loss function
    """
    if loss_name == "Focal":
        criterion = FocalLoss()
    elif loss_name == "CE":
        criterion = CELoss()
    else:
        raise ValueError(f"Loss function {criterion} not supported")
    return criterion
