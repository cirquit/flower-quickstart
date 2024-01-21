import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Tuple, Optional
from collections import OrderedDict
from torch.utils.data import DataLoader
from flwr.common import Scalar, NDArrays
from flwr_datasets import FederatedDataset
from torchvision.transforms import Compose, Normalize, ToTensor

def get_device():
    """CUDA device helper"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(model: nn.Module, testloader: DataLoader) -> Tuple[Scalar, Scalar]:
    """Test the model with the predefined criterion on the test dataset"""
    device = get_device()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader, "Testing"):
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def numpy_to_state_dict(model: nn.Module, parameters: NDArrays) -> OrderedDict[str, Scalar]:
    """Fused the numpy numbers with the model keys to make it loadable via model.load_state_dict()"""
    params_dict = zip(model.state_dict().keys(), parameters)
    return OrderedDict({k:torch.tensor(v) for k, v in params_dict})

def get_cifar10_dataloader(
    partitioners: OrderedDict[str,int], test_train_split: float, node_id: int = None) -> Tuple[Optional[DataLoader], DataLoader]:
    """
    :args partitions (OrderedDict[str,int]): FederatedDataset(partitioners)
    :args test_train_split (float): 0-1 (percent of testsize split)
    :args node_id (Optional[int]): which split to take, it none then full testdataset
    :return (Tuple[Optional[DataLoader], DataLoader]): if split, both dataloader, if not only second testdataloader
    """
    fds = FederatedDataset(dataset="cifar10", partitioners=partitioners)
    if node_id != None:
        partition = fds.load_partition(node_id)
        partition = partition.train_test_split(test_size=test_train_split)

    else:
        partition = fds.load_full("test")
    
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.49139968,0.48215827,0.44653124), (0.24703233,0.24348505,0.26158768))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition = partition.with_transform(apply_transforms)
    if node_id != None:
        trainloader = DataLoader(partition["train"], batch_size=32, shuffle=True)
        testloader = DataLoader(partition["test"], batch_size=32)
    else:
        trainloader = None
        testloader = DataLoader(partition, batch_size=32)
    return trainloader, testloader
