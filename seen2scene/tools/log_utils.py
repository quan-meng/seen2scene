import re
import os
import inspect
from typing import List, Optional
from rich.console import Console
from rich.logging import RichHandler
import logging
import torch.distributed as dist


def zero_rank(func):
    def wrapper(*args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return func(*args, **kwargs)

    return wrapper


class Logger:
    def __init__(self, file_name: str = "log", log_dir: str = None, level="info"):
        self.logger = logging.getLogger(file_name)
        self.logger.propagate = False  # Prevent propagation to root logger

        level = level.upper()
        if level == "INFO":
            level = logging.INFO
        elif level == "DEBUG":
            level = logging.DEBUG
        else:
            raise ValueError(f"Invalid log level: {level}")

        self.level = level

        self.logger.setLevel(level)

        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        class FileNameFormatter(logging.Formatter):
            def format(self, record):
                # Get the caller's filename
                frame = inspect.currentframe()
                # Go up the stack until we find the actual caller
                while frame:
                    # Skip logging-related frames and our logger implementation
                    if (
                        frame.f_code.co_filename != __file__
                        and "logging" not in frame.f_code.co_filename
                        and "rich" not in frame.f_code.co_filename
                    ):
                        record.filename = os.path.basename(frame.f_code.co_filename)
                        break
                    frame = frame.f_back
                if frame is None:
                    record.filename = "unknown"
                return super().format(record)

        # Console handler with rich formatting
        console_handler = RichHandler(rich_tracebacks=True, show_path=False)
        console_handler.setLevel(level)
        console_formatter = FileNameFormatter("%(message)s (%(filename)s)")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler for logging to a file
        if log_dir:
            log_file = os.path.join(log_dir, f"{file_name}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = FileNameFormatter(
                "%(asctime)s - %(levelname)s - %(message)s (%(filename)s)"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        self.console = Console()

    @property
    def is_debug(self) -> bool:
        return self.level == logging.DEBUG

    @zero_rank
    def debug(self, message):
        self.logger.debug(message)

    @zero_rank
    def info(self, message):
        self.logger.info(message)

    @zero_rank
    def warning(self, message):
        self.logger.warning(message)

    @zero_rank
    def error(self, message):
        self.logger.error(message)

    @zero_rank
    def critical(self, message):
        self.logger.critical(message)

    @zero_rank
    def print(self, message):
        self.console.print(message)


def get_logger(file_name: Optional[str] = None, debug: Optional[str] = "", **kwargs):
    if isinstance(debug, str):
        debug = [debug]
    if os.environ.get("DEBUG") in debug:
        level = "DEBUG"
    else:
        level = "INFO"

    logger = Logger(file_name=file_name, level=level, **kwargs)

    return logger


def create_slurm_job_name_from_tyro(
    argv: List[str],
    skip_params: Optional[List[str]] = ["slurm"],
    max_length_per_key: int = 4,
) -> str:
    """Extract key parameters from command line arguments to create a job name.
    Keys are abstracted in two steps:
    1. For "A.B.C" only take "C"
    2. For C, "def<>h<>igkl" -> "dhigk" (max max_length_per_key chars), where '<>' can be '_' or '-'

    Args:
        argv: List of command line arguments
        skip_params: List of parameter substrings to skip in parameter keys
        max_length_per_key: Maximum length for abstracted keys (default: 5)
    """
    # Join all args into a single string for easier regex matching
    cmd_str = " ".join(argv) + ""

    # Updated patterns to ensure values don't contain spaces or colons
    patterns = [
        r" ([\w.]+):([^\s:]+)",  # Pattern 1: matches "key:value" without spaces/colons in value
        r' --([^\s]+)\s+["\']?([^-"\'\s:][^"\'\s:]*?)(?=\s|$)["\']?',  # Pattern 2: matches "--key value"
        r" --([\w.-]+)(?=\s+(?:--|[\w.]+:|$)|$)",  # Pattern 3: matches standalone "--key" flags only
    ]

    values = []
    for pattern in patterns:
        matches = re.finditer(pattern, cmd_str)

        for match in matches:
            if len(match.groups()) == 2:
                key, value = match.groups()
                value = f"_{value[:8]}"
            else:
                key = match.group(1)
                value = ""

            # Skip if key contains any skip parameters
            if any(param in key.lower() for param in skip_params):
                continue

            # First take the last part after dot delimiter
            if "." in key:
                key = key.split(".")[-1]

            # Then abstract if it contains delimiters
            if any(delim in key for delim in ["_", "-"]):
                parts = re.split(r"[_-]", key)
                prefix = "".join(part[0] for part in parts[:-1])
                remaining_space = max_length_per_key - len(prefix)
                suffix = parts[-1][:remaining_space]
                key = prefix + suffix

            values.append(f"{key[:max_length_per_key]}{value}")

    return "_".join(values)
