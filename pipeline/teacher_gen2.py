"""Thinking-teacher answers for 8B, one model load per replica, time-boxed, resumable per part.

usage: teacher_gen2.py PROMPTS.jsonl OUT_DIR REPLICA REPLICAS NPARTS TEACHER STUDENT_TOK DEADLINE_EPOCH
Same token ids as the student: top-20 log-probs (as teacher_gen.py). Other vocabulary (Qwen3.8): the text is re-tokenised
for the student and stored as one-hot targets in the same format; the teacher then gives no soft probabilities.
Stored prompts are always the student's ids ending in '<|im_start|>assistant\n<think>\n'. Exit 3: too slow to be useful.
"""
import json
import os
import sys
import time

import numpy as np

K = 20
THINK = '<think>\n'
MIN_ANSWERS = int(os.environ.get('MIN_ANSWERS', 10000))


def student_ids(text, tok, eos):
    """Student token ids of a teacher's full reply, or None if it is not one finished think block + an answer."""
    if text.startswith('<think>'):
        text = text[len('<think>'):].lstrip('\n')
    text = text.replace('<|im_end|>', '').rstrip()
    head, sep, tail = text.partition('</think>')
    if not sep or '</think>' in tail or '<think>' in tail or not head.strip() or not tail.strip():
        return None
    return tok.encode(text, add_special_tokens=False) + [eos]


def main():
    src, out, rep, reps, nparts, teacher, stok, deadline = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7], float(sys.argv[8])
    os.makedirs(out, exist_ok=True)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    sys.path.insert(0, os.path.expanduser('~/cp/alembic_8b_clone_20261030/src'))
    from distillkit.config import load_config
    SYSTEM = load_config(os.path.expanduser('~/cp/alembic_8b_clone_20261030/configs/rag65_seed42.json')).student.system_prompt
    st, tt = AutoTokenizer.from_pretrained(stok), AutoTokenizer.from_pretrained(teacher)
    same = st.get_vocab() == tt.get_vocab()
    eos = st.convert_tokens_to_ids('<|im_end|>')
    suffix = st.encode(THINK, add_special_tokens=False)
    print(f'teacher {teacher}: same vocabulary {same}', flush=True)
    all_rows = [json.loads(line) for line in open(src)]

    def ids_of(tok, msgs, **kw):
        x = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **kw)
        return list(x['input_ids'] if hasattr(x, 'keys') else x)

    llm = LLM(teacher, tensor_parallel_size=int(os.environ['TP']), max_model_len=16384, gpu_memory_utilization=0.90, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=5632, skip_special_tokens=False, **({'logprobs': K} if same else {}))
    t0, kept = time.time(), 0
    for part in range(rep, nparts, reps):
        if os.path.exists(f'{out}/part{part}.npz'):
            continue
        if time.time() > deadline:
            print('deadline reached', flush=True)
            break
        rows = [r for i, r in enumerate(all_rows) if i % nparts == part]
        msgs = [[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': r['user']}] for r in rows]
        sprompts = [ids_of(st, m) + suffix for m in msgs]  # r27 template ends in assistant\n
        res = llm.generate([{'prompt_token_ids': p if same else ids_of(tt, m)} for p, m in zip(sprompts, msgs)], sp)
        P, O, TI, TL, meta = [], [], [], [], []
        for r, p, g in zip(rows, sprompts, res):
            o = g.outputs[0]
            if o.finish_reason != 'stop' or not o.token_ids:
                continue
            ids = student_ids(o.text, st, eos)
            if ids is None:
                continue
            if same:
                ids = list(o.token_ids)
                ti = np.zeros((len(ids), K), np.int32)
                tl = np.full((len(ids), K), -1e4, np.float16)
                for t, lps in enumerate(o.logprobs):
                    for j, (tid, lp) in enumerate(sorted(lps.items(), key=lambda kv: kv[1].rank)[:K]):
                        ti[t, j], tl[t, j] = tid, max(lp.logprob, -1e4)
            else:
                ti = np.zeros((len(ids), K), np.int32); ti[:, 0] = ids
                tl = np.full((len(ids), K), -1e4, np.float16); tl[:, 0] = 0
            P.append(np.asarray(p, np.int32)); O.append(np.asarray(ids, np.int32)); TI.append(ti); TL.append(tl)
            meta.append({'id': r['id'], 'domain': r.get('domain'), 'text': o.text, 'n_prompt': len(p), 'n_out': len(ids)})
        if not P:
            continue
        po = np.cumsum([0] + [len(x) for x in P]); oo = np.cumsum([0] + [len(x) for x in O])
        np.savez(f'{out}/part{part}.tmp.npz', prompt=np.concatenate(P), out=np.concatenate(O), topk_ids=np.concatenate(TI), topk_lp=np.concatenate(TL), prompt_off=po, out_off=oo)
        with open(f'{out}/part{part}.jsonl', 'w') as f:
            f.writelines(json.dumps(m, ensure_ascii=False) + '\n' for m in meta)
        os.replace(f'{out}/part{part}.tmp.npz', f'{out}/part{part}.npz')
        kept += len(meta)
        el = time.time() - t0
        print(f'part {part}: {len(meta)} of {len(rows)} kept, {int(oo[-1])} tokens; {kept} answers in {el / 60:.0f} min', flush=True)
        if kept and kept / el * max(deadline - t0, 1) * reps < MIN_ANSWERS and el > 2400:
            print('too slow', flush=True)
            sys.exit(3)


if __name__ == '__main__':
    main()
