"""Prepare matched pre-tokenized workloads and HF-cache reference probabilities."""
import argparse, hashlib, json, random
from pathlib import Path
import torch
from transformers import AutoTokenizer
import data
from mcq import build_paper, rms
from heads import build_row, HEADS
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--repo-root',default=str(Path(__file__).resolve().parents[1]))
ap.add_argument('--heads',required=True,help='directory with both KL and both CE checkpoints')
ap.add_argument('--out',required=True,help='benchmark working directory')
a=ap.parse_args()
ROOT=Path(a.repo_root).resolve();JOB=Path(a.out).resolve();HEADS_DIR=Path(a.heads).resolve()
(JOB/'heads').mkdir(parents=True,exist_ok=True)
model='Qwen/Qwen3-8B'
tok=AutoTokenizer.from_pretrained(model,revision=data.revision_for(model))
items=data.mmlu_pro()
krng,crng=random.Random(0),random.Random(0)
krows=[build_paper(tok,x['question'],x['options'],krng) for x in items]
crows=[build_row(tok,x['question'],x['options'],crng) for x in items]
keep=[i for i in range(len(items)) if max(len(krows[i][0]),len(crows[i].ids)) <= 4096]
random.Random(0).shuffle(keep)
keep=keep[:500]
variants={'fundm':('linear','kl',24,HEADS_DIR/'linear_L24_s0_head.pt'),
          'fundm_1t':('comparator','kl',24,HEADS_DIR/'comparator_L24_s0_head.pt'),
          'ce':('linear','ce',36,HEADS_DIR/'ce_linear_original.pt'),
          'ce_1t':('comparator','ce',36,HEADS_DIR/'ce_comparator_original.pt')}
refs={}
for key,(arch,recipe,L,path) in variants.items():
 ck=torch.load(path,weights_only=False,map_location='cpu')
 if recipe=='ce':
  if ck['args']['arch'] != arch: raise ValueError(f'{path}: expected {arch}, found {ck["args"]["arch"]}')
  ck={'state_dict':ck['state'],'arch':arch,'hidden':4096,'layer':L,'recipe':'ce'}
 else: ck={**ck,'recipe':'kl'}
 out=JOB/'heads'/f'{key}_serving.pt';torch.save(ck,out)
 d=torch.load(ROOT/('feats_kd/mmlu_pro_L24.pt' if recipe=='kl' else 'feats/mmlu_pro.pt'),map_location='cpu',weights_only=False)
 byid={str(x):j for j,x in enumerate(d['id'])}
 head=HEADS[arch](4096,128).eval()
 head.load_state_dict({k.removeprefix('head.'):v for k,v in ck['state_dict'].items()})
 ref=[]
 with torch.inference_mode():
  for i in keep[:20]:
   j=byid[str(items[i]['id'])]
   q=d['hq'][j:j+1].float()
   k=(d['hk'][d['off'][j]:d['off'][j+1]] if recipe=='kl' else d['hk'][j]).float()[None]
   order=krows[i][3] if recipe=='kl' else crows[i].order
   assert order==d['order'][j]
   if recipe=='kl': q,k=rms(q),rms(k)
   probs=head.score(q,k,torch.ones(k.shape[:2],dtype=torch.bool)).softmax(-1)[0].tolist()
   ref.append(probs)
 refs[key]=ref
td=torch.load(ROOT/'feats_kd/mmlu_pro_L24.pt',map_location='cpu',weights_only=False)
tbyid={str(x):j for j,x in enumerate(td['id'])}
refs['teacher']=[td['tl'][td['off'][tbyid[str(items[i]['id'])]]:td['off'][tbyid[str(items[i]['id'])]+1]].softmax(-1).tolist() for i in keep[:20]]
rows=[]
for i in keep:
 ids,ans,opts,order=krows[i];c=crows[i]
 rows.append({'id':str(items[i]['id']),'kl':{'prompt_token_ids':ids,'option_positions':opts,'answer_position':ans,'order':order},
              'ce':{'prompt_token_ids':c.ids,'option_positions':c.sep_pos,'answer_position':c.answer_pos,'order':c.order},
              'letter_ids':[tok.encode(' '+x,add_special_tokens=False)[0] for x in 'ABCDEFGHIJ'[:len(order)]]})
report={'model':model,'revision':data.revision_for(model),'seed':0,'set':'mmlu_pro','rows':rows,'hf_reference':refs,
        'sample_id_sha256':hashlib.sha256(json.dumps([r['id'] for r in rows],separators=(',',':')).encode()).hexdigest()}
(JOB/'inputs.json').write_text(json.dumps(report))
print('prepared',len(rows),'rows; lengths',min(len(r['kl']['prompt_token_ids']) for r in rows),max(len(r['kl']['prompt_token_ids']) for r in rows),flush=True)
