"""Released CLM with its authors' embedding/vector caches and native causal vLLM encoding.

Cache contents and prefix blocks are reset between timed passes; allocations
remain. CUDA allocator footprint includes both the encoding worker and the
head/cache process. No HTTP/tokenization is timed.
"""
import argparse,hashlib,json,os,statistics,time,sys
from pathlib import Path
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
import numpy as np
import torch
from vllm import LLM,PoolingParams
import fundm_vllm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/CLM/src'))
from clm.embedder import Embedder,l2
from clm.engine import Engine,DEFAULT_MODEL


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--work-dir',default=str(Path(__file__).resolve().parent))
 ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--vector-cache-mib',type=int,default=64)
 a=ap.parse_args();job=Path(a.work_dir).resolve();d=json.loads((job/'clm-inputs.json').read_text());rows=d['rows']
 cfg=dict(model=d['model'],revision=d['revision'],tokenizer_revision=d['revision'],runner='pooling',dtype='bfloat16',tensor_parallel_size=1,
          max_model_len=4096,max_num_batched_tokens=16384,max_num_seqs=16,enforce_eager=True,enable_prefix_caching=True,
          enable_chunked_prefill=False,async_scheduling=False,gpu_memory_utilization=.2,kv_cache_memory_bytes=32768*2*36*8*128*2,
          seed=0,disable_log_stats=False,worker_extension_cls='fundm_vllm.MemoryWorkerExtension',
          hf_overrides={'architectures':['CLMEncoderForPooling'],'num_hidden_layers':36,'fundm_recipe':'embedding'})
 llm=LLM(**cfg)
 class NativeEmbedder(Embedder):
  def __init__(self):super().__init__(max_tokens=None,batch=1<<30)
  def _fetch(self,texts):
   prompts=[{'prompt_token_ids':d['token_ids'][t]} for t in texts]
   out=llm.encode(prompts,PoolingParams(task='embed'),pooling_task='embed',use_tqdm=False)
   return [l2(o.outputs.data.numpy()) for o in out],sum(len(d['token_ids'][t]) for t in texts)
 eng=Engine(embedder=NativeEmbedder(),checkpoint=d['checkpoint'],device='cuda',action_cache=f'{a.vector_cache_mib}MiB')
 head=eng.heads[DEFAULT_MODEL]
 def reset_cache():
  eng.embedder.cache.clear()
  with eng.arena._lock:
   for pool in eng.arena.pools.values():
    pool.slots.clear();pool.free=list(range(pool.capacity-1,-1,-1));pool.hits=pool.misses=pool.evictions=0
  if not llm.reset_prefix_cache():raise RuntimeError('Could not reset prefix-cache contents')
 def run(indices):
  built=[rows[i] for i in indices];states=[r['state'] for r in built];cands=[t for r in built for t in r['candidates']];tokens=[]
  q=eng._cached(f'{head.namespace}/state',head.proj_dim,states,tokens,head.project_states)
  v=eng._cached(f'{head.namespace}/action',head.proj_dim,cands,tokens,head.project_actions)
  probs=[];j=0
  for i,r in enumerate(built):
   K=len(r['candidates']);probs.append(torch.softmax(head.scale*(v[j:j+K]@q[i]),-1).cpu().tolist());j+=K
  torch.cuda.synchronize();return probs
 def prefix_metrics():
  try:return [{'name':m.name,'value':m.value} for m in llm.get_metrics() if 'prefix_cache' in m.name and hasattr(m,'value')]
  except Exception as e:return {'unavailable':str(e)}
 def memory():
  w=llm.collective_rpc('fundm_memory_probe')[0]
  driver={'cuda_allocated_bytes':torch.cuda.memory_allocated(),'cuda_reserved_bytes':torch.cuda.memory_reserved(),
          'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
          'head_parameter_bytes':sum(p.numel()*p.element_size() for h in [head.state_head,head.action_head] for p in h.parameters())}
  # Neither allocator releases reserved blocks during a timed pass: the peaks coexist.
  return {**w,'cuda_allocated_bytes':w['cuda_allocated_bytes']+driver['cuda_allocated_bytes'],
          'cuda_reserved_bytes':w['cuda_reserved_bytes']+driver['cuda_reserved_bytes'],
          'cuda_peak_reserved_bytes':w['cuda_reserved_bytes']+driver['cuda_reserved_bytes'],
          'cuda_peak_allocated_bytes':w['cuda_peak_allocated_bytes']+driver['cuda_peak_allocated_bytes'],
          'encoder_worker':w,'head_cache_process':driver,'model_parameter_bytes':w['model_parameter_bytes']+driver['head_parameter_bytes']}
 reset_cache();probs=run(list(range(20)));refs=[r['reference'] for r in rows[:20]]
 errs=[abs(x-y) for p,q in zip(probs,refs) for x,y in zip(p,q)];agree=sum(max(range(len(p)),key=p.__getitem__)==max(range(len(q)),key=q.__getitem__) for p,q in zip(probs,refs))/20
 validation={'n':20,'argmax_agreement':agree,'max_probability_error':max(errs),'mean_probability_error':statistics.mean(errs)}
 print('VALIDATION CLM',validation,flush=True)
 if agree<.95 or statistics.mean(errs)>.01:raise ValueError('Native CLM differs materially from released-Engine reference')
 reset_cache();run(list(range(500)));reset_cache()
 llm.collective_rpc('fundm_memory_probe',args=(True,));torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 repeats=[];mem=[]
 for rep in range(a.repeats):
  reset_cache();before=prefix_metrics();t=time.perf_counter();run(list(range(500)));elapsed=time.perf_counter()-t
  r={'seconds':elapsed,'questions_per_second':500/elapsed,'cache':eng.arena.stats(),'prefix_before':before,'prefix_after':prefix_metrics()}
  repeats.append(r);mem.append(memory());print('THROUGHPUT CLM',rep,r,flush=True)
 throughput_memory=max(mem,key=lambda x:x['cuda_peak_reserved_bytes'])
 reset_cache();run([0]);reset_cache();llm.collective_rpc('fundm_memory_probe',args=(True,));torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 latency=[]
 for i in range(50):
  t=time.perf_counter();run([i]);latency.append({'id':rows[i]['id'],'milliseconds':1000*(time.perf_counter()-t)})
 latency_memory=memory()
 out={'variant':'clm','recipe':'released CLM','layers':36,'n':500,'repeats':a.repeats,'engine_config':cfg,
      'sample_id_sha256':d['sample_id_sha256'],'validation':validation,'throughput':repeats,'throughput_median':statistics.median(r['questions_per_second'] for r in repeats),
      'latency':latency,'first_output_median_ms':statistics.median(r['milliseconds'] for r in latency),'memory_throughput':throughput_memory,'memory_latency':latency_memory,
      'latency_cache':eng.arena.stats(),'vector_cache_mib':a.vector_cache_mib,'checkpoint_sha256':d['checkpoint_sha256'],
      'versions':{'vllm':__import__('vllm').__version__,'torch':torch.__version__},
      'source_hashes':{f:hashlib.sha256((Path(__file__).resolve().parent/f).read_bytes()).hexdigest() for f in ['fundm_vllm.py','serve_clm_benchmark.py']}}
 (job/'results/clm.json').write_text(json.dumps(out,indent=2));print('RESULT CLM',out['throughput_median'],out['first_output_median_ms'],throughput_memory,flush=True)

if __name__=='__main__':main()
