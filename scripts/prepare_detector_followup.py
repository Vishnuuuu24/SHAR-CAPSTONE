#!/usr/bin/env python3
"""Validate owner-supplied SigLIP weights and unpack RWF locally; run inside a pod."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        w = args.data_root / '.cache/weights/siglip-base-patch16-224-full'
        sha = digest(w / 'model.safetensors')
        expected = '2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8'
        if sha != expected:
            raise ValueError(f'SigLIP checksum differs from registered upstream file: {sha}')
        print('SigLIP upstream checksum verified', flush=True)
        from safetensors.torch import load_file
        import torch
        full = load_file(str(w / 'model.safetensors'))
        vision = load_file(str(args.data_root / '.cache/weights/siglip_base_patch16_224_vision-f5d84933.safetensors'))
        # transformers 5 uses a flat vision dictionary, while the full model prefixes its tower.
        matched = 0
        for key, value in vision.items():
            candidate = key if key.startswith('vision_model.') else 'vision_model.' + key
            if candidate not in full or not torch.equal(value, full[candidate]):
                raise ValueError(f'Full SigLIP vision tensor differs from cached-feature tower: {key}')
            matched += 1
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(w / 'tokenizer.json'))
        tokenizer.enable_padding(length=64, pad_id=1, pad_token='</s>')
        tokenizer.enable_truncation(max_length=64)
        token = tokenizer.encode('a surveillance video of normal activity')
        assert len(token.ids) == 64 and max(token.ids) < 32000
        print(f'Vision tensors match exactly ({matched}); supplied tokenizer.json works', flush=True)
        del full, vision
        archive = args.data_root / 'Datasets/RWF-2000.tar.gz'
        if not archive.exists():
            archive = args.data_root / 'Datasets/RWF-2000/download/RWF-2000.tar.gz'
        archive_sha = digest(archive)
        destination = args.data_root / 'Datasets/RWF-2000'
        stage = args.output / 'extracted'
        stage.mkdir()
        files, counts = [], Counter()
        with tarfile.open(archive, 'r|gz') as tf:
            for member in tf:
                parts = Path(member.name).parts
                if member.isdir():
                    continue
                if not member.isfile() or len(parts) != 4 or parts[0] != 'RWF-2000' or parts[1] not in ('train', 'val') or parts[2] not in ('Fight', 'NonFight') or Path(parts[3]).suffix.lower() != '.avi':
                    raise ValueError(f'Unexpected archive entry: {member.name}')
                safe_name = hashlib.sha256(member.name.encode('utf-8', errors='surrogateescape')).hexdigest() + '.avi'
                target = stage.joinpath(parts[1], parts[2], safe_name)
                target.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                with tf.extractfile(member) as src, target.open('xb') as dst:
                    for block in iter(lambda: src.read(1024 * 1024), b''):
                        h.update(block)
                        dst.write(block)
                if target.stat().st_size != member.size:
                    raise ValueError(f'Truncated member: {member.name}')
                files.append({'path': str(Path(parts[1], parts[2], safe_name)), 'archive_member': member.name, 'bytes': member.size, 'sha256': h.hexdigest()})
                counts['/'.join(parts[1:3])] += 1
                if len(files) % 200 == 0:
                    print(json.dumps({'extracted': len(files), 'seconds': round(time.time()-started, 1)}), flush=True)
        if dict(counts) != {'train/Fight': 800, 'train/NonFight': 800, 'val/Fight': 200, 'val/NonFight': 200}:
            raise ValueError(f'Unexpected split counts: {counts}')
        digests = {split: {f['sha256'] for f in files if f['path'].startswith(split+'/')} for split in ('train', 'val')}
        overlap = sorted(digests['train'] & digests['val'])
        if overlap:
            print(f'WARNING: {len(overlap)} exact cross-split video duplicates; recorded, native split retained', flush=True)
        for split in ('train', 'val'):
            target = destination / split
            if target.exists():
                raise FileExistsError(f'Refusing to overwrite {target}; staged files retained at {stage}')
        for split in ('train', 'val'):
            shutil.move(str(stage / split), str(destination / split))
        report = {'status': 'completed', 'siglip_sha256': sha, 'vision_tensors_equal': matched,
                  'tokenizer': type(tokenizer).__name__, 'archive': str(archive), 'archive_sha256': archive_sha,
                  'counts': dict(counts), 'exact_train_val_duplicates': overlap, 'videos': files,
                  'seconds': time.time()-started, 'scope': 'input intake only; no model training or test scoring'}
        (args.output / 'intake.json').write_text(json.dumps(report, indent=1)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k != 'videos'}), flush=True)
    except Exception as exc:
        (args.output / 'failure.json').write_text(json.dumps({'error': repr(exc), 'seconds': time.time()-started})+'\n')
        raise


if __name__ == '__main__':
    main()
