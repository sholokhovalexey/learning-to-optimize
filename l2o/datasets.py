"""Problem-agnostic dataset contracts and generic collate helpers."""

from __future__ import annotations

import torch.utils.data
from abc import ABC, abstractmethod
from typing import Generic, TypeVar


TItem = TypeVar("TItem")


class BaseTaskDataset(torch.utils.data.Dataset, ABC, Generic[TItem]):
    """Base class for problem-specific task datasets."""

    @abstractmethod
    def __getitem__(self, idx: int) -> TItem:
        """Return one task/sample consumed by problem-specific collate functions."""
        raise NotImplementedError

