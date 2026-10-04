"""Run the original E7 statistical cells on the public historical artifacts.

Only notebook path configuration is changed. Bootstrap/sign-flip settings and
statistical code remain those of the September model-conditional analysis.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile


def safe_extract(archive, destination):
    for item in archive.infolist():
        target = (destination / item.filename).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError('Unsafe archive path')
    archive.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root', type=Path, default=Path('.'))
    parser.add_argument('--destination', type=Path, default=Path('rerun_historical_statistics'))
    args = parser.parse_args()
    root = args.repository_root.resolve()
    destination = args.destination.resolve()
    input_dir = destination / 'inputs'
    input_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / 'historical_results.zip') as archive:
        for stem in ['E3_RESULTS_FINAL.zip', 'E4_RESULTS_FINAL.zip', 'E6_RESULTS_FINAL.zip']:
            candidates = [n for n in archive.namelist() if n.endswith('/' + stem)]
            assert len(candidates) == 1, stem
            value = archive.read(candidates[0])
            with zipfile.ZipFile(io.BytesIO(value)) as nested:
                # Validate paths before the unchanged notebook extracts its inputs.
                for name in nested.namelist():
                    assert (input_dir / name).resolve().is_relative_to(input_dir.resolve()), name
            (input_dir / stem).write_bytes(value)
    notebook_path = root / 'historical_notebooks/07_E7_Cluster_Aware_Statistical_Inference_CalibBERT_v2.ipynb'
    notebook = json.loads(notebook_path.read_text())
    namespace = {'__name__': '__main__', 'destination': destination, 'input_dir': input_dir,
                 'display': lambda *args, **kwargs: None}
    config = ''.join(notebook['cells'][1]['source'])
    config = config.replace('USE_GOOGLE_DRIVE = True', 'USE_GOOGLE_DRIVE = False')
    config = config.replace('Path("/content/calibbert_v2_e7")', 'destination')
    config = config.replace('INPUT_DIR = Path("/content")', 'INPUT_DIR = input_dir')
    exec(compile(config, str(notebook_path) + ':configuration', 'exec'), namespace)
    executed = []
    for index in range(3, 30):
        cell = notebook['cells'][index]
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            exec(compile(source, str(notebook_path) + f':cell{index}', 'exec'), namespace)
            executed.append({'index': index, 'source_sha256': hashlib.sha256(source.encode()).hexdigest()})
    report = {'scope': 'Original September E7 model-conditional analysis; contemporary five-seed predictions are excluded',
              'notebook': str(notebook_path.relative_to(root)), 'executed_original_statistical_cells': executed,
              'N_BOOT': namespace['N_BOOT'], 'N_SIGNFLIP': namespace['N_SIGNFLIP'],
              'RNG_SEED': namespace['RNG_SEED'], 'configuration_changes': 'Filesystem paths and Drive mounting only',
              'public_E6_archive': 'Representative_Docs removed; numerical statistical inputs unchanged'}
    (destination / 'PUBLIC_RERUN_SCOPE.json').write_text(json.dumps(report, indent=2))
    print('HISTORICAL STATISTICS COMPLETE:', destination / 'outputs_e7')


if __name__ == '__main__':
    main()
