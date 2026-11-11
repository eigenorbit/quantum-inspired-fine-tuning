"""Print where the data, the outputs and the pretrained weights are (`repro.settings`).

    python -m repro.where                    # every location and the settings file
    python -m repro.where DERMA_REPRO_OUT    # one value (exit code 1 if not set)

The scripts in trichoscopy/scripts and experiments/*/scripts ask this module
instead of repeating the rules.
"""

from __future__ import annotations

import os
import sys

from . import settings


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    settings.apply()
    if argv:
        try:
            print(settings.require(argv[0]))
        except settings.LocationError as exc:
            print(exc, file=sys.stderr)
            return 1
        return 0
    f = settings.config_file()
    print(f"settings file: {f} ({'found' if f.is_file() else 'absent'})")
    for key in settings.KEYS:
        print(f"  {key:22s} {os.environ.get(key, '(not set)')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
