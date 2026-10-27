# SPDX-License-Identifier: Apache-2.0
"""Dependency spike only: not a trashpanda trainer or its eventual application tests.
Run from repo root: .venv/bin/python -m pytest planning/first-model/test_compatibility.py -q
Requires the three pinned Mistral tokenizer JSON files in .venv/compat-tokenizer.
"""
import copy
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

torch.set_num_threads(1)
TOKENIZER = Path(__file__).resolve().parents[2] / '.venv/compat-tokenizer'


def config(**overrides):
    values = dict(vocab_size=32, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                  max_position_embeddings=32, rope_theta=10000., rms_norm_eps=1e-5,
                  tie_word_embeddings=True, attention_bias=False, mlp_bias=False,
                  attention_dropout=0., initializer_range=.02, bos_token_id=1,
                  eos_token_id=2, pad_token_id=None, use_cache=False)
    values.update(overrides)
    return LlamaConfig(**values)


def model(**overrides):
    torch.manual_seed(42)
    result = LlamaForCausalLM(config(**overrides))
    assert result.config._attn_implementation == 'sdpa'
    return result


def test_exact_production_parameter_count():
    with torch.device('meta'):
        m = model(vocab_size=32000, hidden_size=1024, intermediate_size=2560,
                  num_hidden_layers=22, num_attention_heads=16, num_key_value_heads=16,
                  max_position_embeddings=2048)
    assert sum(p.numel() for p in m.parameters()) == 298103808
    assert m.lm_head.weight is m.model.embed_tokens.weight


def test_pinned_tokenizer_special_strings_and_local_roundtrip(tmp_path):
    t = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    assert len(t) == 32000
    assert (t.unk_token_id, t.bos_token_id, t.eos_token_id) == (0, 1, 2)
    s = '<s> hello </s> <unk> café 熊'
    ids = t.encode(s, add_special_tokens=False, split_special_tokens=True)
    assert not ({0, 1, 2} & set(ids))
    assert t.decode(ids) == s
    assert t.encode('', add_special_tokens=False, split_special_tokens=True) == []
    t.save_pretrained(tmp_path)
    again = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert again.encode(s, add_special_tokens=False, split_special_tokens=True) == ids


def test_causal_sdpa_matches_eager_and_shifted_loss():
    m = model().eval()
    x = torch.tensor([[3, 4, 5, 6, 7, 8]])
    changed = x.clone(); changed[:, 4:] = 10
    with torch.no_grad():
        a = m(x, labels=x)
        b = m(changed)
    torch.testing.assert_close(a.logits[:, :4], b.logits[:, :4], atol=1e-6, rtol=1e-5)
    manual = F.cross_entropy(a.logits[:, :-1].reshape(-1, 32), x[:, 1:].reshape(-1))
    torch.testing.assert_close(a.loss, manual, atol=1e-6, rtol=1e-5)
    cfg = config(); cfg._attn_implementation = 'eager'
    eager = LlamaForCausalLM(cfg).eval(); eager.load_state_dict(m.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(a.logits, eager(x).logits, atol=1e-6, rtol=1e-5)


def test_accumulation_matches_full_batch():
    a = model(); b = copy.deepcopy(a)
    x = torch.tensor([[3,4,5,6], [7,8,9,10], [3,3,4,4], [5,5,6,6]])
    a(x, labels=x).loss.backward()
    for part in x.chunk(2):
        (b(part, labels=part).loss / 2).backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, atol=1e-6, rtol=1e-4)


def test_tiny_model_learns():
    m = model(); opt = torch.optim.AdamW(m.parameters(), lr=.01)
    x = torch.tensor([[3,4,5,6]*4]*2)
    initial = m(x, labels=x).loss.item()
    for _ in range(25):
        opt.zero_grad(set_to_none=True)
        loss = m(x, labels=x).loss; loss.backward(); opt.step()
    final = m(x, labels=x).loss.item()
    assert final < initial * .5, (initial, final)


def test_native_optimizer_scheduler_rng_checkpoint_roundtrip(tmp_path):
    # Proves dependency primitives, not the future trainer's data cursor or atomicity.
    data = torch.arange(8 * 16).reshape(8, 16) % 27 + 3
    def setup():
        m = model(); opt = torch.optim.AdamW(m.parameters(), lr=.001)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: 1.0 / (n + 1))
        generator = torch.Generator().manual_seed(123)
        return m, opt, sched, generator
    def step(m, opt, sched, generator):
        ids = torch.randperm(8, generator=generator)[:2]
        opt.zero_grad(set_to_none=True)
        x = data[ids]; m(x, labels=x).loss.backward(); opt.step(); sched.step()
        return ids.tolist()
    full = setup(); order = [step(*full) for _ in range(6)]
    split = setup(); first = [step(*split) for _ in range(3)]
    p = tmp_path / 'state.pt'
    torch.save(dict(model=split[0].state_dict(), optimizer=split[1].state_dict(),
                    scheduler=split[2].state_dict(), generator=split[3].get_state(),
                    rng=torch.get_rng_state(), step=3), p)
    restored = setup(); state = torch.load(p, weights_only=True)
    restored[0].load_state_dict(state['model']); restored[1].load_state_dict(state['optimizer'])
    restored[2].load_state_dict(state['scheduler']); restored[3].set_state(state['generator'])
    torch.set_rng_state(state['rng'])
    after = [step(*restored) for _ in range(state['step'], 6)]
    assert order == first + after
    for a, b in zip(full[0].parameters(), restored[0].parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert full[2].state_dict() == restored[2].state_dict()


def test_standard_hf_export_offline_fresh_process(tmp_path):
    t = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    m = model(vocab_size=32000).eval()
    x = t('hello world', return_tensors='pt', add_special_tokens=False).input_ids
    with torch.no_grad():
        expected = m(x).logits
    m.save_pretrained(tmp_path, safe_serialization=True); t.save_pretrained(tmp_path)
    torch.save({'input': x, 'expected': expected}, tmp_path / 'reference.pt')
    code = '''
import sys,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
torch.set_num_threads(1)
p=sys.argv[1]
t=AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False)
m=AutoModelForCausalLM.from_pretrained(p,local_files_only=True,trust_remote_code=False).eval()
r=torch.load(p+'/reference.pt',weights_only=True)
with torch.no_grad():
 torch.testing.assert_close(m(r['input']).logits,r['expected'],atol=1e-6,rtol=1e-5)
 out=m.generate(r['input'],attention_mask=torch.ones_like(r['input']),max_new_tokens=3,do_sample=False,pad_token_id=t.eos_token_id)
assert out.shape[1]>r['input'].shape[1]
print('offline native HF generation passed')
'''
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], env=env,
                            capture_output=True, text=True, check=True)
    assert 'passed' in result.stdout
    assert (tmp_path / 'model.safetensors').exists()
    assert not list(tmp_path.glob('*.py'))
