import torch
from torch.utils.data import DataLoader, TensorDataset

__all__ = ['MyDataLoader', 'PreFetcher']


class PreFetcher:
    r""" Data pre-fetcher to accelerate the data loading
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.len = len(loader)
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.next_input = None

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            for idx, tensor in enumerate(self.next_input):
                self.next_input[idx] = tensor.cuda(non_blocking=True)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.loader = iter(self.ori_loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        if input is None:
            raise StopIteration
        for tensor in input:
            tensor.record_stream(torch.cuda.current_stream())
        self.preload()
        return input


class MyDataLoader(object):
    r""" PyTorch DataLoader for COST2100 dataset.
    """

    def __init__(self, train_path, val_path, test_path, batch_size,
                 num_workers, pin_memory, channel=2, nt=32, nc=32,
                 return_indices=False,
                 train_tensor=None, val_tensor=None, test_tensor=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.channel = channel
        self.nt = nt
        self.nc = nc
        self.return_indices = return_indices
        
        if train_tensor is not None:
            train_tensor = self._validate_tensor(train_tensor)
            self.train_dataset = self._build_dataset(train_tensor)
        else:
            train_tensor = self._load_tensor(train_path)
            self.train_dataset = self._build_dataset(train_tensor)
        if val_tensor is not None:
            val_tensor = self._validate_tensor(val_tensor)
            self.val_dataset = self._build_dataset(val_tensor)
        else:
            val_tensor = self._load_tensor(val_path)
            self.val_dataset = self._build_dataset(val_tensor)
        if test_tensor is not None:
            test_tensor = self._validate_tensor(test_tensor)
            self.test_dataset = self._build_dataset(test_tensor)
        else:
            test_tensor = self._load_tensor(test_path)
            self.test_dataset = self._build_dataset(test_tensor)

    def _validate_tensor(self, data):
        expected_shape = (self.channel, self.nt, self.nc)
        if data.ndim == 2:
            data = data.view(-1, *expected_shape)
        if data.ndim != 4 or tuple(data.shape[1:]) != expected_shape:
            raise ValueError(
                f"tensor should have shape (N, {self.channel}, "
                f"{self.nt}, {self.nc}), got {tuple(data.shape)}")
        return data.to(torch.float32)

    def _load_tensor(self, path):
        data = torch.load(path, weights_only=True,
                          map_location=torch.device('cpu'))
        data = data.to(torch.float32)
        expected_shape = (self.channel, self.nt, self.nc)
        if data.ndim == 2:
            data = data.view(-1, *expected_shape)
        if data.ndim != 4 or tuple(data.shape[1:]) != expected_shape:
            raise ValueError(
                f"{path} should have shape (N, {self.channel}, "
                f"{self.nt}, {self.nc}), got {tuple(data.shape)}")
        return data

    def _build_dataset(self, data):
        if not self.return_indices:
            return TensorDataset(data)
        indices = torch.arange(data.size(0), dtype=torch.long)
        return TensorDataset(data, indices)


    def __call__(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  shuffle=True)
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                shuffle=False)
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=self.pin_memory,
                                 shuffle=False)

        # Accelerate CUDA data loading with pre-fetcher if GPU is used.
        if self.pin_memory is True:
            train_loader = PreFetcher(train_loader)
            val_loader = PreFetcher(val_loader)
            test_loader = PreFetcher(test_loader)

        return train_loader, val_loader, test_loader
