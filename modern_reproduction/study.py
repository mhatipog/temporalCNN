"""Exploratory TempCNN public-example study, not the published full-data experiment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import example


def read_predictions(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    return (np.array([int(x['polygon_id']) for x in rows]),
            np.array([int(x['true_class']) for x in rows]),
            np.array([int(x['predicted_class']) for x in rows]))


def scores(y, pred, groups):
    matrix = np.zeros((13, 13), dtype=int)
    np.add.at(matrix, (y, pred), 1)
    true_count = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    diag = np.diag(matrix)
    recall = np.divide(diag, true_count, out=np.zeros(13), where=true_count > 0)
    f1 = np.divide(2 * diag, true_count + predicted_count, out=np.zeros(13),
                   where=(true_count + predicted_count) > 0)
    polygon_correct = []
    for group in np.unique(groups):
        idx = groups == group
        if len(set(y[idx])) != 1:
            raise ValueError('A test polygon has inconsistent class labels')
        polygon_correct.append(int(np.argmax(np.bincount(pred[idx], minlength=13)) == y[idx][0]))
    return {
        'accuracy': float(np.mean(y == pred)),
        'macro_recall': float(recall.mean()),
        'macro_f1': float(f1.mean()),
        'polygon_majority_accuracy': float(np.mean(polygon_correct)),
        'recall_by_class': recall.tolist(),
    }


def nearest_neighbor(train_x, train_y, test_x):
    train = train_x.reshape(len(train_x), -1).astype('float64')
    test = test_x.reshape(len(test_x), -1).astype('float64')
    guesses = []
    for start in range(0, len(test), 64):
        block = test[start:start + 64]
        dist = np.sum((block[:, None] - train[None]) ** 2, axis=2)
        guesses.extend(train_y[np.argmin(dist, axis=1)].tolist())
    return np.array(guesses, dtype=int)


def paired_polygon_interval(groups, y, first, second, seed, draws=4000):
    """Descriptive cluster bootstrap conditional on the two fitted models."""
    rng = np.random.default_rng(seed)
    identifiers = np.unique(groups)
    sizes = np.array([int((groups == p).sum()) for p in identifiers])
    differences = np.array([
        int((second[groups == p] == y[groups == p]).sum() -
            (first[groups == p] == y[groups == p]).sum())
        for p in identifiers
    ])
    which = rng.integers(len(identifiers), size=(draws, len(identifiers)))
    bootstrap = differences[which].sum(axis=1) / sizes[which].sum(axis=1)
    return np.quantile(bootstrap, [0.025, 0.5, 0.975]).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, nargs='+', default=[7, 42, 101])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--output', type=Path, default=Path('study-results'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        'claim': 'Exploratory sensitivity analysis on small public example, not paper reproduction',
        'seeds': args.seeds, 'runs': [], 'paired_differences': [],
        'limitations': [
            'Full published research dataset is not public',
            'Small validation groups; estimates vary across random splits',
            'Bootstrap covers held-out test polygon sampling conditional on trained models only',
            'Class-balanced split and patience differ from the published protocol',
        ],
    }
    for seed in args.seeds:
        predictions = {}
        for protocol in ('pre-split', 'train-only'):
            target = args.output / str(seed) / protocol
            report = example.run(argparse.Namespace(
                data='example', output=str(target), seed=seed, protocol=protocol,
                validation_split='balanced-20pct', epochs=args.epochs, patience=args.patience))
            polygons, y, prediction = read_predictions(target / 'predictions.csv')
            train_x, labels, _, _, test_x, _, _, _, _, _ = example.prepare(
                Path('example/train_dataset.csv'), Path('example/test_dataset.csv'),
                seed, protocol, 'balanced-20pct')
            control = nearest_neighbor(train_x, labels, test_x)
            report['classification_metrics'] = scores(y, prediction, polygons)
            report['one_nearest_neighbor_control'] = scores(y, control, polygons)
            result['runs'].append({
                'seed': seed, 'protocol': protocol,
                'epochs_completed': report['epochs_completed'],
                'neural': report['classification_metrics'],
                'one_nearest_neighbor': report['one_nearest_neighbor_control'],
                'validation_class_counts': report['validation_class_counts'],
                'normalization_p2': report['normalization_p2'],
                'normalization_p98': report['normalization_p98'],
            })
            predictions[protocol] = prediction
        result['paired_differences'].append({
            'seed': seed,
            'train_only_minus_pre_split': float(np.mean(predictions['train-only'] == y) -
                                                np.mean(predictions['pre-split'] == y)),
            'test_polygon_cluster_bootstrap_95pct': paired_polygon_interval(
                polygons, y, predictions['pre-split'], predictions['train-only'], seed),
        })
        (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['paired_differences'], indent=2))


if __name__ == '__main__':
    main()
