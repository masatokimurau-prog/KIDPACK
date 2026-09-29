"""Software version information recorded with every data file."""
import subprocess
from pathlib import Path

from kidpack import __version__


def software_version():
    """Return {'kidpack': version, 'git': 'abc1234[-dirty]' or None}."""
    info = {'kidpack': __version__, 'git': None}
    try:
        out = subprocess.run(['git', 'describe', '--always', '--dirty'],
                             cwd=Path(__file__).parent, capture_output=True,
                             text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return info
    if out.returncode == 0:
        info['git'] = out.stdout.strip()
    return info
