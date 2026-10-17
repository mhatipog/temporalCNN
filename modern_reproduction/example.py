"""Re-run the published TempCNN example, not the paper's full benchmark.

Requires KERAS_BACKEND=torch (Keras 3 + PyTorch). The existing example CSVs
in the parent repository are the only inputs. Original entry points are untouched.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np

BLOBS = {
    'train_dataset.csv': '989888be989b897b27d5e6e10afabaebcaee8cdd',
    'test_dataset.csv': 'a53fd8207b9cfec27d1e9d3ad2b8d91d672c37e7',
}


def load(path):
    a = np.loadtxt(path, delimiter=',', ndmin=2)
    if a.shape[1] != 449 or len(a) != 260 or not np.isfinite(a).all():
        raise ValueError(f'Unexpected TempCNN example dataset: {path} {a.shape}')
    if not (a[:, :2] == a[:, :2].astype(int)).all():
        raise ValueError('Non-integer labels or polygon identifiers')
    return a[:, 2:].reshape(-1, 149, 3).astype('float32'), a[:, 1].astype(int), a[:, 0].astype(int)


def git_blob(path):
    data = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def split(polygons, seed=42, fraction=0.05):
    ids = list(np.unique(polygons))
    random.Random(seed).shuffle(ids)
    held = set()
    for value in ids[:-1]:
        held.add(int(value))
        if np.count_nonzero(np.isin(polygons, list(held))) >= round(len(polygons) * fraction):
            break
    val = np.flatnonzero(np.isin(polygons, list(held)))
    train = np.flatnonzero(~np.isin(polygons, list(held)))
    if len(val) == 0 or len(train) == 0:
        raise ValueError('Empty split')
    return train, val, sorted(held)


def balanced_split(polygons, labels, seed=42, per_class=4):
    """Select exactly four observations per class, keeping polygons intact.

    An exploratory split for the small public example, not the paper's split.
    """
    groups = {}
    for polygon, label in zip(polygons, labels):
        polygon, label = int(polygon), int(label)
        if polygon in groups and groups[polygon][0] != label:
            raise ValueError('Polygon contains multiple labels')
        if polygon in groups:
            groups[polygon] = (label, groups[polygon][1] + 1)
        else:
            groups[polygon] = (label, 1)

    rng = random.Random(seed)
    held = []
    for label in np.unique(labels):
        candidates = [(pid, n) for pid, (cls, n) in groups.items() if cls == label]
        rng.shuffle(candidates)
        choices = {0: []}
        for pid, n in candidates:
            for count, ids in list(choices.items()):
                new_count = count + n
                if new_count <= per_class and new_count not in choices:
                    choices[new_count] = ids + [pid]
        if per_class not in choices:
            raise ValueError(f'Cannot hold out {per_class} whole-polygon observations of class {label}')
        held.extend(choices[per_class])
    val = np.flatnonzero(np.isin(polygons, held))
    train = np.flatnonzero(~np.isin(polygons, held))
    if set(polygons[train]) & set(polygons[val]):
        raise AssertionError('Polygon leakage across train and validation')
    assert len(val) == per_class * len(np.unique(labels))
    return train, val, sorted(held)


def normalizer(x):
    low, high = np.percentile(x, [2, 98], axis=(0, 1))
    if not np.all(high > low):
        raise ValueError('Constant channel')
    return low, high


def prepare(train_csv, test_csv, seed, protocol, validation_split):
    x, polygons, labels = load(train_csv)
    xt, test_polygons, yt = load(test_csv)
    if np.intersect1d(polygons, test_polygons).size:
        raise ValueError('Polygon overlap between train and test')
    if not np.array_equal(np.unique(labels), np.arange(13)):
        raise ValueError('Expected 13 classes')
    t, v, val_ids = (balanced_split(polygons, labels, seed) if validation_split == 'balanced-20pct' else split(polygons, seed))
    if len(np.unique(labels[t])) != 13:
        raise ValueError('Validation split removed a class from training')
    lo, hi = normalizer(x[t] if protocol == 'train-only' else x)
    norm = lambda z: ((z - lo) / (hi - lo)).astype('float32')
    return (norm(x[t]), labels[t], norm(x[v]), labels[v],
            norm(xt), yt, test_polygons, val_ids, lo, hi)


def model():
    import keras
    from keras import layers
    from keras.regularizers import l2
    inputs = keras.Input((149, 3))
    z = inputs
    for _ in range(3):
        z = layers.Conv1D(64, 5, padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-6))(z)
        z = layers.BatchNormalization()(z)
        z = layers.Activation('relu')(z)
        z = layers.Dropout(0.5)(z)
    z = layers.Flatten()(z)
    z = layers.Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(1e-6))(z)
    z = layers.BatchNormalization()(z)
    z = layers.Activation('relu')(z)
    z = layers.Dropout(0.5)(z)
    out = layers.Dense(13, activation='softmax', kernel_regularizer=l2(1e-6))(z)
    result = keras.Model(inputs, out)
    result.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return result


def run(args):
    os.environ.setdefault('KERAS_BACKEND', 'torch')
    import keras
    import torch
    keras.utils.set_random_seed(args.seed)
    torch.set_num_threads(2)
    data = Path(args.data)
    identity = {k: git_blob(data / k) for k in BLOBS}
    if identity != BLOBS:
        raise ValueError(f'Inputs differ from published example CSVs: {identity}')
    tr, y, va, yv, te, yt, test_poly, val_ids, lo, hi = prepare(
        data / 'train_dataset.csv', data / 'test_dataset.csv', args.seed, args.protocol, args.validation_split)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    net = model()
    history = net.fit(tr, y, validation_data=(va, yv), epochs=args.epochs,
                      batch_size=32, verbose=2, shuffle=True, callbacks=[
        keras.callbacks.ModelCheckpoint(str(out / 'best.keras'), monitor='val_loss',
                                        save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience),
    ])
    net = keras.models.load_model(out / 'best.keras')
    predictions = np.argmax(net.predict(te, verbose=0), axis=1)
    matrix = np.zeros((13, 13), dtype=int)
    np.add.at(matrix, (yt, predictions), 1)
    np.savetxt(out / 'confusion_matrix.csv', matrix, fmt='%d', delimiter=',')
    with (out / 'predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['polygon_id', 'true_class', 'predicted_class'])
        writer.writerows(zip(test_poly, yt, predictions))
    result = {
        'claim': 'Public example-data run; not replication of published full-data results',
        'upstream_commit': 'd081da2f6eef19f36b5517549eab3bbbedd18598',
        'input_git_blobs': identity,
        'model': 'Archi_3CONV64_1FC256',
        'protocol': args.protocol, 'seed': args.seed,
        'validation_split': args.validation_split, 'early_stopping_patience': args.patience,
        'validation_class_counts': np.bincount(yv, minlength=13).tolist(),
        'training_class_counts': np.bincount(y, minlength=13).tolist(),
        'validation_polygon_ids': val_ids,
        'samples': {'train': len(y), 'validation': len(yv), 'test': len(yt)},
        'normalization_p2': lo.tolist(), 'normalization_p98': hi.tolist(),
        'epochs_completed': len(history.history['loss']),
        'correct': int(np.trace(matrix)), 'total': int(matrix.sum()),
        'accuracy': float(np.trace(matrix) / matrix.sum()),
        'train_history': {k: [float(v) for v in values] for k, values in history.history.items()},
        'runtime': {'python': platform.python_version(), 'keras': keras.__version__,
                    'torch': torch.__version__, 'numpy': np.__version__},
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'protocol': args.protocol, 'accuracy': result['accuracy'],
                      'correct': result['correct'], 'test': result['total']}))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='example')
    p.add_argument('--output', required=True)
    p.add_argument('--protocol', choices=['train-only', 'pre-split'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--validation-split', choices=['original-5pct', 'balanced-20pct'],
                   default='balanced-20pct')
    run(p.parse_args())
