#!/usr/bin/env python3
"""Export a real NIH training example and genuine AlphaRAD token/attention data.

Uses existing frozen features and the NIH-validation-selected Pre MLP, seed 42.
No fitting or example selection based on scores/attention is performed.
Run with AlphaRAD/local_setup/python_local.sh from the medical workspace.
"""
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT))
from experiments.large_patient_multilabel.common import load_protocol, load_data
from experiments.baseline_token_classification.common import sha256, save, residual_mlp
from experiments.direct_anchor_classification.model import FrozenClassificationHead

RUN = ROOT / 'experiments/large_patient_multilabel/runs/20261121'


def read(path):
    return json.loads(Path(path).read_text())


def native_attention(head, tokens):
    """Exact FLaS attention: eight heads, native scale, softmax includes CLS."""
    assert head.kind == 'alpharad' and head.config['include_cls']
    values = F.normalize(tokens.reshape(len(tokens), tokens.shape[1], head.heads, -1), dim=-1)
    queries = F.normalize(head.queries.reshape(len(head.queries), head.heads, -1), dim=-1)
    weights = (torch.einsum('bkhd,qhd->bqkh', values, queries) * head.config['scale']).softmax(2)
    context = torch.einsum('bqkh,bkhd->bqhd', weights, values).flatten(-2)
    torch.testing.assert_close(context, head.context(tokens), atol=1e-6, rtol=1e-6)
    # Native FP32 softmax sums 1,370 entries; retain its values, allowing the
    # accumulated rounding error without renormalizing the displayed attention.
    total = weights.sum(2, dtype=torch.float64)
    torch.testing.assert_close(total, torch.ones_like(total), atol=5e-6, rtol=0)
    mean = weights[0].mean(-1)
    return mean[:, 1:].reshape(14, 37, 37).numpy(), mean[:, 0].numpy()


def main():
    torch.set_num_threads(4)
    protocol = load_protocol(RUN)
    meta, rows = load_data()
    wanted = [meta['class_names'].index(c) for c in ('Atelectasis', 'Effusion')]
    candidates = sorted([(i, r) for i, r in enumerate(rows)
        if r['split'] == 'train' and all(r['labels'][c] for c in wanted)], key=lambda pair: pair[1]['id'])
    index, row = candidates[0]
    assert sha256(row['image']) == row['file_sha256']
    out = RUN / 'features/AlphaRAD'
    cache = read(out / 'cache.json')
    for path, digest in cache['compact_sha256'].items():
        assert sha256(path) == digest
    integrity = read(RUN / 'source_integrity.json')
    spec = next(s for s in cache['token_shards'] if index in s['indices'])
    local = spec['indices'].index(index)
    stat = Path(spec['path']).stat()
    assert (stat.st_size, stat.st_mtime_ns) == (integrity[spec['path']]['bytes'], integrity[spec['path']]['mtime_ns'])
    array = np.load(spec['path'], mmap_mode='r')
    assert list(array.shape) == spec['shape']
    tokens = torch.from_numpy(np.array(array[local:local+1], copy=True))
    assert tokens.shape == (1, 1370, 768)
    config = read(out / 'head.json')
    head = FrozenClassificationHead(config).eval().requires_grad_(False)
    seal = read(RUN / 'SELECTION_LOCK.json')
    selection = next(r for r in seal['selections'] if (r['model'], r['method'], r['seed']) == ('AlphaRAD', 'pre_mlp', 42))
    assert sha256(selection['checkpoint']) == selection['checkpoint_sha256']
    payload = torch.load(selection['checkpoint'], map_location='cpu', weights_only=True)
    assert payload['head_config'] == config and payload['classes'] == meta['class_names']
    assert payload['protocol_sha256'] == sha256(RUN / 'PROTOCOL.json')
    adapter = residual_mlp(768, 64).eval().requires_grad_(False)
    adapter.load_state_dict(payload['module_state_dict'], strict=True)
    with torch.inference_mode():
        adapted = adapter(tokens)
        original_scores = head(tokens).numpy()[0]
        adapted_scores = head(adapted).numpy()[0]
        np.testing.assert_allclose(original_scores, np.load(out / 'native.npy', mmap_mode='r')[index], atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(head.context(tokens).numpy()[0],
            np.load(out / 'contexts.npy', mmap_mode='r')[index], atol=3e-5, rtol=3e-5)
        original_attention, original_cls = native_attention(head, tokens)
        adapted_attention, adapted_cls = native_attention(head, adapted)
        torch.testing.assert_close(residual_mlp(768, 64).eval()(tokens), tokens, atol=0, rtol=0)
    # Reproduce the encoder's actual deterministic image geometry. No contrast
    # enhancement, anatomical crop, synthesized image, or horizontal flip.
    base = torch.load(cache['checkpoint_path'], map_location='cpu', weights_only=False)
    transform_config = base['config']['transform']['test']
    assert transform_config == {'Resize': {'size': 518}, 'CenterCrop': {'size': 518}}
    del base
    gray = cv2.imread(row['image'], cv2.IMREAD_GRAYSCALE)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    image = transforms.Compose([transforms.ToPILImage(), transforms.Resize(518), transforms.CenterCrop(518)])(rgb)
    image_array = np.asarray(image)
    token_indices = np.linspace(1, 1369, 16, dtype=int)
    channels = np.linspace(0, 767, 16, dtype=int)
    token_matrix = tokens[0].numpy()[np.ix_(token_indices, channels)]
    assets = PAPER / 'figures/token_mlp_real_assets.npz'
    np.savez_compressed(assets, image_rgb=image_array, tokens=tokens[0].numpy(), adapted_tokens=adapted[0].numpy(),
        token_matrix=token_matrix, displayed_token_indices=token_indices, displayed_channels=channels,
        queries=head.queries.numpy(), class_names=meta['class_names'], labels=row['labels'],
        original_scores=original_scores, adapted_scores=adapted_scores,
        original_attention=original_attention, adapted_attention=adapted_attention,
        original_cls_attention=original_cls, adapted_cls_attention=adapted_cls)
    evidence = dict(status='verified', model='AlphaRAD', seed=42, source_run=str(RUN.relative_to(ROOT)),
        selection_rule='Lexicographically first NIH TRAIN image labeled Atelectasis and Effusion; independent of scores/maps.',
        sample_index=index, image_id=row['id'], source_image=row['image'], image_sha256=row['file_sha256'],
        split=row['split'], positive_dataset_labels=[c for c, y in zip(meta['class_names'], row['labels']) if y],
        checkpoint_sha256=selection['checkpoint_sha256'], selected_epoch=selection['best_epoch'],
        head_sha256=sha256(out / 'head.json'), protocol_sha256=sha256(RUN / 'PROTOCOL.json'),
        source_manifest_sha256=meta['samples_sha256'], source_token_file=spec['path'], source_token_local_index=local,
        displayed_sample_tokens_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
        encoder_geometry=transform_config, preprocessing='Native grayscale-to-RGB, bilinear Resize(518), CenterCrop(518).',
        attention='Exact native softmax over CLS + 1369 patches, averaged over eight FLaS heads; CLS omitted only from the spatial display; no patch renormalization.',
        native_score_max_abs_error=float(np.abs(original_scores-np.load(out / 'native.npy', mmap_mode='r')[index]).max()),
        validation=['Native context/logit agreement with verified cached features', 'Exact reconstructed native attention/context agreement',
                    'Attention sums to one including CLS', 'Source image and selected adapter SHA256', 'Zero-initialized adapter equals identity'],
        interpretation='One training example illustrating computation; not localization validation or an estimate of classification improvement.',
        exported_assets_sha256=sha256(assets), exporter_sha256=sha256(Path(__file__)))
    save(PAPER / 'supplement_data/token_mlp_figure_provenance.json', evidence)
    print('Exported real image and native attention:', row['id'], 'NIH train;', selection['best_epoch'], 'selected epoch.')


if __name__ == '__main__':
    main()
