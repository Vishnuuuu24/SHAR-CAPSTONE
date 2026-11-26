"""Frozen SigLIP prompt features in the exact cached pooler space; no new image pass."""
from pathlib import Path
import json
import numpy as np

ACTIVITIES = ('normal activity', 'abuse', 'arson', 'assault', 'burglary', 'an explosion',
              'fighting', 'a road accident', 'robbery', 'shooting', 'shoplifting', 'stealing',
              'vandalism', 'an arrest')
TEMPLATES = ('a surveillance video of {}.', 'a security camera recording of {}.')


def token_ids(folder, texts):
    """Use the supplied fast tokenizer directly; do not require a missing SentencePiece model."""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(folder)/'tokenizer.json'))
    tokenizer.enable_truncation(max_length=64)
    tokenizer.enable_padding(length=64,pad_id=1,pad_token='</s>')
    return np.asarray([e.ids for e in tokenizer.encode_batch([s.lower() for s in texts])],dtype=np.int64)


def encode_prompts(folder, device):
    import torch
    from safetensors.torch import load_file
    from transformers import SiglipConfig, SiglipTextModel
    folder = Path(folder)
    config = SiglipConfig.from_dict(json.loads((folder/'config.json').read_text())).text_config
    model = SiglipTextModel(config).to(device).eval()
    weights = load_file(str(folder/'model.safetensors'),device='cpu')
    selected = {k[len('text_model.'):]:v for k,v in weights.items() if k.startswith('text_model.')}
    # Transformers changed the standalone tower's dictionary prefix across releases.
    if set(selected) != set(model.state_dict()):
        selected = {'text_model.'+k:v for k,v in selected.items()}
    model.load_state_dict(selected,strict=True)
    del weights,selected
    texts = [t.format(a) for a in ACTIVITIES for t in TEMPLATES]
    ids = torch.from_numpy(token_ids(folder,texts)).to(device)
    with torch.no_grad():
        embeddings = model(input_ids=ids).pooler_output.float()
        embeddings = torch.nn.functional.normalize(embeddings,dim=-1)
        embeddings = embeddings.reshape(len(ACTIVITIES),len(TEMPLATES),-1).mean(1)
        embeddings = torch.nn.functional.normalize(embeddings,dim=-1).cpu().numpy()
    return embeddings, {'activities':list(ACTIVITIES),'templates':list(TEMPLATES),'pooling':'L2 normalize each template, mean templates, L2 normalize'}


def similarities(poolers, embeddings):
    x = np.asarray(poolers,dtype=np.float32)
    norms = np.linalg.norm(x,axis=-1,keepdims=True)
    if np.any(norms == 0):
        raise ValueError('zero SigLIP pooler vector')
    return (x/norms) @ np.asarray(embeddings,dtype=np.float32).T
