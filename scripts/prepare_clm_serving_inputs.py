import argparse,json,hashlib,sys
from pathlib import Path
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import data
ap=argparse.ArgumentParser();ap.add_argument('--work-dir',required=True);ap.add_argument('--repo-root',default=str(Path(__file__).resolve().parents[1]));a=ap.parse_args()
job=Path(a.work_dir).resolve();root=Path(a.repo_root).resolve()
sys.path.insert(0,str(root/'third_party/CLM/src'))
from clm.schema import build_pairs
parent=json.loads((job/'inputs.json').read_text());byid={it['id']:it for it in data.mmlu_pro()}
tok=AutoTokenizer.from_pretrained(parent['model'],revision=parent['revision'])
refs={str(x['question_id']):x['p_clm'] for x in map(json.loads,(root/'results/clm_mmlu_pro_preds.jsonl').open())}
rows=[];alltext={}
for row in parent['rows']:
 it=byid[row['id']];q={'type':'choice','instructions':None,'criteria':{str(i):o for i,o in enumerate(it['options'])}}
 state,keys,cands=build_pairs(it['question'],{'q':q})['q']
 rows.append({'id':it['id'],'state':state,'candidates':cands,'reference':refs[it['id']]})
 for t in [state,*cands]:alltext.setdefault(t,tok.encode(t,add_special_tokens=False))
ck=hf_hub_download('Contrastive-LM/CLM-v0.1-8B','CLM_v0.1-8B.pt',revision=data.revision_for('Contrastive-LM/CLM-v0.1-8B'))
out={'model':parent['model'],'revision':parent['revision'],'sample_id_sha256':parent['sample_id_sha256'],'rows':rows,'token_ids':alltext,'checkpoint':ck,'checkpoint_sha256':hashlib.sha256(Path(ck).read_bytes()).hexdigest()}
(job/'clm-inputs.json').write_text(json.dumps(out));print('prepared',len(rows),'questions',len(alltext),'unique texts',flush=True)
