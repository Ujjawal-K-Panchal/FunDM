"""Matched native-vLLM prefix inference: throughput, first decision, whole CUDA allocator footprint.

Fresh process per variant. Tokenized inputs exclude tokenizer/HTTP overhead. All five variants use the native pooling engine and return option probabilities.
The teacher applies its original full-vocabulary LM head at the answer position;
FunDM heads project the selected states. All backbones remain causal.
Memory includes weights, actual KV tensors, activations and CUDA allocator slack;
it excludes non-PyTorch allocations/context and is not discrete-GPU NVML VRAM.
"""
import argparse, hashlib, json, os, statistics, time
from pathlib import Path
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
from vllm import LLM, PoolingParams
import fundm_vllm  # registers the prefix architecture and memory worker extension


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['teacher','fundm','fundm_1t','ce','ce_1t'],required=True)
 ap.add_argument('--work-dir',required=True,help='prepared workload/checkpoint directory')
 ap.add_argument('--smoke',action='store_true');ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--limit',type=int,default=500)
 a=ap.parse_args();job=Path(a.work_dir).resolve();d=json.loads((job/'inputs.json').read_text())
 rows=d['rows'][:a.limit];teacher=a.variant=='teacher';ce=a.variant in ('ce','ce_1t');L=36 if teacher or ce else 24
 recipe='ce' if ce else 'kl'
 config=dict(model=d['model'],revision=d['revision'],tokenizer_revision=d['revision'],dtype='bfloat16',tensor_parallel_size=1,
             max_model_len=4096,max_num_batched_tokens=16384,max_num_seqs=16,enforce_eager=True,
             enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,gpu_memory_utilization=0.2,
             kv_cache_memory_bytes=32768*2*L*8*128*2,seed=0,worker_extension_cls='fundm_vllm.MemoryWorkerExtension')
 config['runner']='pooling';config['hf_overrides']={'architectures':['FunDMForPooling'],'num_hidden_layers':L,'num_labels':10,
  'fundm_head_path':None if teacher else str(job/'heads'/f'{a.variant}_serving.pt'),'fundm_recipe':'teacher' if teacher else recipe}
 engine=LLM(**config)
 def run(indices):
  prompts=[{'prompt_token_ids':rows[i][recipe]['prompt_token_ids']} for i in indices]
  if teacher:
   params=[PoolingParams(task='classify',use_activation=False,extra_kwargs={'answer_position':rows[i][recipe]['answer_position'],'letter_ids':rows[i]['letter_ids']}) for i in indices]
  else:
   params=[PoolingParams(task='classify',use_activation=False,extra_kwargs={k:rows[i][recipe][k] for k in ('option_positions','answer_position')}) for i in indices]
  return engine.classify(prompts,pooling_params=params,use_tqdm=False)
 n=min(20,len(rows));outputs=run(list(range(n)))
 validation={}
 if not teacher:
  probs=[list(o.outputs.probs[:len(rows[i][recipe]['option_positions'])]) for i,o in enumerate(outputs)]
  refs=d['hf_reference'][a.variant][:n]
  errors=[abs(float(p)-q) for ps,qs in zip(probs,refs) for p,q in zip(ps,qs)]
  agreement=sum(max(range(len(p)),key=p.__getitem__)==max(range(len(q)),key=q.__getitem__) for p,q in zip(probs,refs))/n
  validation={'n':n,'argmax_agreement':agreement,'max_probability_error':max(errors),'mean_probability_error':statistics.mean(errors),'probs':probs}
  print('VALIDATION',a.variant,json.dumps({k:v for k,v in validation.items() if k!='probs'}),flush=True)
  if agreement < .95 or max(errors) > .05: raise ValueError('Native vLLM adapter differs materially from HF cached-state reference')
 if teacher:
  refs=d['hf_reference']['teacher'][:n]
  tprobs=[list(o.outputs.probs[:len(rows[i]['letter_ids'])]) for i,o in enumerate(outputs)]
  slots=[max(range(len(p)),key=p.__getitem__) for p in tprobs]
  agreement=sum(s==max(range(len(p)),key=p.__getitem__) for s,p in zip(slots,refs))/n
  errors=[abs(x-y) for p,q in zip(tprobs,refs) for x,y in zip(p,q)]
  validation={'n':n,'argmax_agreement':agreement,'max_probability_error':max(errors),'mean_probability_error':statistics.mean(errors)}
  tie_agreement=sum(q[slot]==max(q) for slot,q in zip(slots,refs))/n
  validation['tie_aware_argmax_agreement']=tie_agreement
  validation['exact_reference_top_ties']=sum(sum(v==max(q) for v in q)>1 for q in refs)
  validation['probs']=tprobs
  print('VALIDATION teacher',{k:v for k,v in validation.items() if k!='probs'},flush=True)
  (job/'teacher-validation.json').write_text(json.dumps({'validation':validation,'reference':refs},indent=2))
  if tie_agreement < .95 or statistics.mean(errors) > .01: raise ValueError('Teacher letter readout differs materially from HF reference (exact HF ties permitted)')
 if a.smoke:
  print('MEMORY',engine.collective_rpc("fundm_memory_probe"),flush=True);return
 # A complete untimed workload pass establishes a warm engine; prefix-cache reuse is disabled.
 run(list(range(len(rows))))
 engine.collective_rpc("fundm_memory_probe",args=(True,))
 repeats=[]
 for rep in range(a.repeats):
  t=time.perf_counter();run(list(range(len(rows))));elapsed=time.perf_counter()-t
  repeats.append({'seconds':elapsed,'questions_per_second':len(rows)/elapsed})
  print('THROUGHPUT',a.variant,rep,repeats[-1],flush=True)
 throughput_memory=engine.collective_rpc("fundm_memory_probe")[0]
 # Sequential batch-one latencies are first-output latency, not queueing latency under load.
 latency_indices=list(range(min(50,len(rows))));run([latency_indices[0]])
 engine.collective_rpc("fundm_memory_probe",args=(True,))
 latency=[]
 for i in latency_indices:
  t=time.perf_counter();run([i]);latency.append({'id':rows[i]['id'],'milliseconds':1000*(time.perf_counter()-t)})
 latency_memory=engine.collective_rpc("fundm_memory_probe")[0]
 result={'variant':a.variant,'recipe':recipe,'layers':L,'n':len(rows),'repeats':a.repeats,'engine_config':config,
         'sample_id_sha256':d['sample_id_sha256'],'validation':validation,'throughput':repeats,
         'throughput_median':statistics.median(r['questions_per_second'] for r in repeats),
         'latency':latency,'first_output_median_ms':statistics.median(r['milliseconds'] for r in latency),
         'memory_throughput':throughput_memory,'memory_latency':latency_memory,
         'versions':{'vllm':__import__('vllm').__version__,'torch':__import__('torch').__version__},
         'source_hashes':{**{f:hashlib.sha256((Path(__file__).resolve().parent/f).read_bytes()).hexdigest() for f in ['fundm_vllm.py','serve_benchmark.py']},'inputs.json':hashlib.sha256((job/'inputs.json').read_bytes()).hexdigest()}}
 (job/'results').mkdir(exist_ok=True);(job/'results'/f'{a.variant}.json').write_text(json.dumps(result,indent=2))
 print('RESULT',a.variant,result['throughput_median'],result['first_output_median_ms'],throughput_memory,flush=True)

if __name__=='__main__':main()
