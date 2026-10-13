import os
import pathlib
import sys

def validate(path):
    path = pathlib.Path(path).resolve()
    if len(path.parts) < 4 or path.parts[1] != 'Volumes':
        raise SystemExit('An external SSD workspace is required.')
    mount = pathlib.Path('/Volumes') / path.parts[2]
    ancestor = path
    while not ancestor.exists() and ancestor != mount:
        ancestor = ancestor.parent
    if not os.path.ismount(mount) or not os.access(ancestor, os.W_OK):
        raise SystemExit('The external SSD is not mounted and writable. No fallback is permitted.')
    if any(part in ('Documents', 'Desktop', 'Mobile Documents') for part in path.parts):
        raise SystemExit('The project must be outside iCloud-synced folders.')
    return path


if __name__ == '__main__':
    print('Verified external workspace:', validate(sys.argv[1]))
