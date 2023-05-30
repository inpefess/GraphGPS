import os
from typing import Callable, List, Optional, Tuple, Union

import torch
from pyg_proto import pyg_pb2
from torch_geometric.data import Data, InMemoryDataset


def indices_from_file(filename: str) -> List[int]:
    with open(filename, "r") as input_file:
        lines = input_file.readlines()
    return [int(line[:-1]) for line in lines]


class TPTPDataset(InMemoryDataset):
    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
    ):
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.data, self.slices = torch.load(self.processed_paths[0])
        split_indices = torch.load(self.processed_paths[1])
        self.split_idxs = [
            split_indices["train"],
            split_indices["valid"],
            split_indices["test"],
        ]

    def _get_graphs(self, labels: List[int]) -> List[Data]:
        graphs = []
        for raw_filename in os.listdir(os.path.join(self.raw_dir, "graphs")):
            data_proto = pyg_pb2.Data()
            with open(
                os.path.join(self.raw_dir, "graphs", raw_filename), "rb"
            ) as raw_file:
                raw_protobuf = raw_file.read()
            data_proto.ParseFromString(raw_protobuf)
            data_item = Data(
                x=torch.Tensor(data_proto.x).view(-1, 1),
                edge_index=torch.LongTensor(
                    [data_proto.edge_index_source, data_proto.edge_index_target]
                ),
                edge_attr=torch.Tensor(data_proto.edge_attr).view(-1, 1),
                y=labels[int(os.path.splitext(raw_filename)[0])],
            )
            graphs.append(data_item)
        return graphs

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        labels = indices_from_file(os.path.join(self.raw_dir, "labels.txt"))
        graphs = self._get_graphs(labels)
        torch.save(self.collate(graphs), self.processed_paths[0])
        split_indices = {
            "train": indices_from_file(os.path.join(self.raw_dir, "train.txt")),
            "valid": indices_from_file(os.path.join(self.raw_dir, "valid.txt")),
            "test": indices_from_file(os.path.join(self.raw_dir, "test.txt")),
        }
        torch.save(split_indices, self.processed_paths[1])

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing."""
        return ["data.pt", "split_dict.pt"]
