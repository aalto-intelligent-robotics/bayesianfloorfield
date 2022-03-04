from os import PathLike
from typing import Any, Optional, Union

from numpy import ndarray

def loadmat(
    file_name: Union[str, bytes, PathLike],
    mdict: Optional[dict] = None,
    appendmat: bool = True,
    **kwargs: Any
) -> dict[str, ndarray]: ...
