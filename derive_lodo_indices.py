"""Reconstruct September LODO indices from its frozen split and domain taxonomy."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile
import numpy as np
import pandas as pd


def reconstruct(source_root, historical_archive, destination):
    config = json.loads((source_root / 'protocol.json').read_text())
    for name, key in [('dreaddit-train.csv', 'train_sha256'), ('dreaddit-test.csv', 'test_sha256'),
                      ('development_split_assignment.csv', 'split_sha256')]:
        assert hashlib.sha256((source_root / 'inputs' / name).read_bytes()).hexdigest() == config[key], name
    train = pd.read_csv(source_root / 'inputs/dreaddit-train.csv', dtype={'id': str, 'post_id': str})
    test = pd.read_csv(source_root / 'inputs/dreaddit-test.csv', dtype={'id': str, 'post_id': str})
    split = pd.read_csv(source_root / 'inputs/development_split_assignment.csv', dtype={'id': str, 'post_id': str})
    with zipfile.ZipFile(historical_archive) as archive:
        prefix = next(n.rsplit('/', 1)[0] for n in archive.namelist()
                      if n.endswith('/Results_Expanded/E4_ANALYSIS_LOCK.json'))
        lock = json.loads(archive.read(prefix + '/E4_ANALYSIS_LOCK.json'))
        manifest = pd.read_csv(io.BytesIO(archive.read(prefix + '/e4_lodo_fold_manifest.csv')))
        predictions = pd.read_csv(io.BytesIO(archive.read(prefix + '/e4_prediction_level_long.csv')),
                                  dtype={'id': str, 'post_id': str})
        taxonomy = lock['domain_taxonomy']
        # The archived E4 mapping is subreddit -> domain.
        train['domain'] = train.subreddit.map(taxonomy)
        test['domain'] = test.subreddit.map(taxonomy)
        assert train.domain.notna().all() and test.domain.notna().all()
        assert np.array_equal(train.iloc[split.row_index].id.to_numpy(), split.id.to_numpy())
        assert np.array_equal(train.iloc[split.row_index].post_id.to_numpy(), split.post_id.to_numpy())
        split = split.copy()
        split['domain'] = train.iloc[split.row_index].domain.to_numpy()
        parts = []
        roles = ['source_train', 'source_calibration', 'target_test', 'source_reference_test']
        for heldout in lock['domains']:
            selections = [
                ('train', train, split.loc[split.split.isin(['fit', 'validation']) & split.domain.ne(heldout), 'row_index'].to_numpy()),
                ('train', train, split.loc[split.split.eq('calibration') & split.domain.ne(heldout), 'row_index'].to_numpy()),
                ('test', test, np.flatnonzero(test.domain.eq(heldout))),
                ('test', test, np.flatnonzero(test.domain.ne(heldout))),
            ]
            reference = manifest.set_index('heldout_domain').loc[heldout]
            for role, (dataset, frame, index) in zip(roles, selections):
                selected = frame.iloc[index][['id', 'post_id', 'label', 'domain']].copy()
                assert len(selected) == int(reference[role + '_n'])
                assert abs(selected.label.mean() - reference[role + '_stress_rate']) < 1e-12
                selected.insert(0, 'row_index', index)
                selected.insert(0, 'role', role)
                selected.insert(0, 'dataset_split', dataset)
                selected.insert(0, 'heldout_domain', heldout)
                parts.append(selected)
            for condition in ['Vanilla', 'LS0.10']:
                name = prefix + f'/E4_LODO_{heldout}_{condition}_seed42_logits.npz'
                logits = np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
                for i, field in [(1, 'cal_y'), (2, 'target_y'), (3, 'source_y')]:
                    _, frame, index = selections[i]
                    assert np.array_equal(frame.iloc[index].label.to_numpy(), logits[field]), (name, field)
                for i, group in [(2, 'target_OOD'), (3, 'source_reference')]:
                    _, frame, index = selections[i]
                    expected = frame.iloc[index].sort_values('id').reset_index(drop=True)
                    saved = predictions.loc[predictions.heldout_domain.eq(heldout)
                                            & predictions.condition.eq(condition)
                                            & predictions.evaluation_group.eq(group)].sort_values('id').reset_index(drop=True)
                    assert len(expected) == len(saved)
                    for column in ['id', 'post_id']:
                        assert expected[column].equals(saved[column]), (heldout, condition, group, column)
                    assert np.array_equal(expected.label.to_numpy(), saved.true_label.to_numpy())
        out = pd.concat(parts, ignore_index=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(destination, index=False)
        report = {'scope': 'Reconstructed from frozen September E4 notebook rules, not an original saved index file',
                  'frozen_raw_input_hashes_verified': True, 'all_fold_counts_and_class_rates_match': True,
                  'all_10_fold_logit_files_calibration_target_source_label_arrays_match': True,
                  'all_target_source_prediction_identities_match': True,
                  'rows': len(out), 'heldout_domains': lock['domains'],
                  'csv_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}
        destination.with_suffix('.json').write_text(json.dumps(report, indent=2))
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('contemporary_multiseed/source'))
    parser.add_argument('--historical-archive', type=Path, default=Path('historical_results.zip'))
    parser.add_argument('--destination', type=Path, default=Path('rerun_lodo_indices.csv'))
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.source_root, args.historical_archive, args.destination), indent=2))
