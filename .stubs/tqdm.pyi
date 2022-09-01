from io import StringIO, TextIOWrapper
from typing import Any, Iterable, Iterator, Optional, Union

class tqdm:
    def __init__(
        self,
        iterable: Optional[Iterable] = None,
        desc: Optional[str] = None,
        total: Optional[Union[int, float]] = None,
        leave: bool = True,
        file: Optional[Union[TextIOWrapper, StringIO]] = None,
        ncols: Optional[int] = None,
        mininterval: float = 0.1,
        maxinterval: float = 10.0,
        miniters: Optional[Union[int, float]] = None,
        ascii: Optional[Union[bool, str]] = None,
        disable: bool = False,
        unit: str = "it",
        unit_scale: Union[bool, int, float] = False,
        dynamic_ncols: bool = False,
        smoothing: float = 0.3,
        bar_format: Optional[str] = None,
        initial: Union[int, float] = 0,
        position: Optional[int] = None,
        postfix: Optional[dict] = None,
        unit_divisor: float = 1000.0,
        write_bytes: Optional[bool] = None,
        lock_args: Optional[tuple] = None,
        nrows: Optional[int] = None,
        colour: Optional[str] = None,
        delay: float = 0.0,
        gui: bool = False,
        **kwargs: Any
    ) -> None: ...
    def __iter__(self) -> Iterator: ...

def trange(*args: Any, **kwargs: Any) -> tqdm: ...
