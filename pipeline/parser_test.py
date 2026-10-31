import sys,os; sys.path.insert(0,os.path.expanduser('~/cp/8b/bin'))
import teacher_gen2 as g
from transformers import AutoTokenizer
st=AutoTokenizer.from_pretrained(os.path.expanduser('~/cp/models/r27')); eos=st.convert_tokens_to_ids('<|im_end|>')
for s,w in [('<think>\nplan\n</think>\n\nans<|im_end|>',True),('plan\n</think>\n\nans',True),('plan only',False),('a</think>b</think>c',False),('</think>\n\nans',False),('plan</think>',False)]:
    r=g.student_ids(s,st,eos); print(repr(s),'ok' if (r is not None)==w else 'WRONG', repr(st.decode(r)) if r else None)
r=g.student_ids('plan\n</think>\n\nans',st,eos); print('ends eos',r[-1]==eos,'think token kept',st.convert_tokens_to_ids('</think>') in r)
