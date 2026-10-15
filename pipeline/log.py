"""Console output of a pipeline run: numbered steps, with details and timing indented under them."""
import sys
import time
from contextlib import contextmanager

from tqdm import tqdm

INDENT = ' ' * 6


def duration(seconds):
    return f'{seconds:.2f} s'


@contextmanager
def step(index, total, title):
    """Print '[index/total] title' and, when the step ends, how long it took."""
    print(f'[{index}/{total}] {title}', flush=True)
    start = time.time()
    yield
    detail(f'done in {duration(time.time() - start)}')


def progress(iterable, unit, total=None):
    """An indented progress bar over iterable, shown only in a terminal."""
    return tqdm(iterable, desc=INDENT + unit, total=total, disable=not sys.stderr.isatty())


def detail(text):
    print(INDENT + text, flush=True)


def warning(text):
    print(f'{INDENT}warning: {text}', flush=True)
