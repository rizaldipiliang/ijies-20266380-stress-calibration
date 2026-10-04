"""Stage byte-identical Dreaddit inputs without changing the frozen protocol."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('contemporary_multiseed/source'))
    parser.add_argument('--archive', type=Path, help='Original Dreaddit ZIP downloaded from its authors')
    parser.add_argument('--train', type=Path)
    parser.add_argument('--test', type=Path)
    args = parser.parse_args()
    config = json.loads((args.source_root / 'protocol.json').read_text())
    requested = [('dreaddit-train.csv', 'train_sha256', args.train),
                 ('dreaddit-test.csv', 'test_sha256', args.test)]
    if bool(args.train) != bool(args.test) or (not args.archive and not args.train):
        parser.error('Supply --archive, or both --train and --test.')
    staged = {}
    for name, key, explicit in requested:
        if explicit:
            candidates = [(str(explicit), explicit.read_bytes())]
        else:
            with zipfile.ZipFile(args.archive) as archive:
                candidates = [(n, archive.read(n)) for n in archive.namelist()
                              if Path(n).name == name]
        matches = [(origin, value) for origin, value in candidates if digest(value) == config[key]]
        if not matches:
            raise ValueError(f'{name}: no byte-identical input matches {config[key]}. '
                             'Keep the frozen protocol and split unchanged; resolve the source-file mismatch.')
        staged[name] = matches[0][1]
    inputs = args.source_root / 'inputs'
    if digest((inputs / 'development_split_assignment.csv').read_bytes()) != config['split_sha256']:
        raise ValueError('The frozen development split checksum does not match.')
    inputs.mkdir(parents=True, exist_ok=True)
    for name, value in staged.items():
        target = inputs / name
        if target.exists() and target.read_bytes() != value:
            raise FileExistsError(f'Refusing to replace a different input: {target}')
        target.write_bytes(value)
    print('INPUT CHECKSUMS PASS. Official train/test bytes and archived split are unchanged.')


if __name__ == '__main__':
    main()
