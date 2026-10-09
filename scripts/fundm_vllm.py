"""Native vLLM pooling adapter: frozen Qwen3 prefix and the trained FunDM head."""
import re
import torch
from torch import nn
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.layers.pooler.abstract import Pooler
from heads import HEADS


class RawFinalState(nn.Module):
    def forward(self, hidden_states, residual=None):
        return (hidden_states if residual is None else hidden_states + residual), None


class DecisionPooler(Pooler):
    def __init__(self, checkpoint):
        super().__init__()
        self.head = HEADS[checkpoint['arch']](checkpoint['hidden'],128).float()
        self.head.load_state_dict({k.removeprefix('head.'):v for k,v in checkpoint['state_dict'].items()})
        self.head.eval()
        self.recipe = checkpoint.get('recipe','kl')

    def get_supported_tasks(self):
        return {'classify'}

    def forward(self, hidden_states, pooling_metadata):
        cursor = pooling_metadata.get_pooling_cursor()
        if cursor.is_partial_prefill():
            raise RuntimeError('FunDM requires complete prefill; disable chunking and prefix-cache reads')
        B = len(pooling_metadata.pooling_params)
        positions, answers = [], []
        for i, params in enumerate(pooling_metadata.pooling_params):
            extra = params.extra_kwargs or {}
            end = int(cursor.prompt_lens_cpu[i])-1
            positions.append(extra.get('option_positions',[end]))
            answers.append(extra.get('answer_position',end))
        K = max(map(len,positions))
        index = torch.zeros((B,K),dtype=torch.long,device=hidden_states.device)
        mask = torch.zeros_like(index,dtype=torch.bool)
        for i,p in enumerate(positions):
            index[i,:len(p)] = torch.tensor(p,device=index.device) + cursor.first_token_indices_gpu[i]
            mask[i,:len(p)] = True
        qindex = cursor.first_token_indices_gpu + torch.tensor(answers,device=index.device)
        q,k = hidden_states[qindex].float(),hidden_states[index].float()
        if self.recipe == 'kl':
            q = q * torch.rsqrt(q.square().mean(-1,keepdim=True)+1e-6)
            k = k * torch.rsqrt(k.square().mean(-1,keepdim=True)+1e-6)
        logits = self.head.score(q,k,mask)
        probs = torch.softmax(logits,-1)
        result = torch.zeros((B,10),device=probs.device,dtype=torch.float32)
        result[:,:K] = probs
        return result


class TeacherLetterPooler(Pooler):
    """Full model's original LM head, evaluated at the answer position."""
    def __init__(self, config):
        super().__init__()
        self.lm_head = nn.Linear(config.hidden_size,config.vocab_size,bias=False,dtype=torch.bfloat16)

    def get_supported_tasks(self):
        return {'classify'}

    def forward(self, hidden_states, pooling_metadata):
        cursor=pooling_metadata.get_pooling_cursor()
        if cursor.is_partial_prefill(): raise RuntimeError('Teacher requires complete prefill')
        extra=[p.extra_kwargs or {} for p in pooling_metadata.pooling_params]
        positions=[x.get('answer_position',int(cursor.prompt_lens_cpu[i])-1) for i,x in enumerate(extra)]
        q=hidden_states[cursor.first_token_indices_gpu+torch.tensor(positions,device=hidden_states.device)]
        logits=self.lm_head(q).float()  # original full-vocabulary projection
        result=torch.zeros((len(extra),10),device=logits.device,dtype=torch.float32)
        for i,x in enumerate(extra):
            ids=x.get('letter_ids',[362])
            result[i,:len(ids)]=logits[i,ids].softmax(-1)
        return result


class EncoderStatePooler(Pooler):
    """CLM's independently encoded, L2-normalized final-token states."""
    def get_supported_tasks(self): return {'embed'}
    def forward(self, hidden_states, pooling_metadata):
        cursor=pooling_metadata.get_pooling_cursor()
        return torch.nn.functional.normalize(hidden_states[cursor.last_token_indices_gpu].float(),dim=-1)


class FunDMForPooling(nn.Module):
    is_pooling_model = True
    default_seq_pooling_type = 'ALL'
    default_tok_pooling_type = 'ALL'
    attn_type = 'decoder'
    packed_modules_mapping = {'qkv_proj':['q_proj','k_proj','v_proj'],'gate_up_proj':['gate_proj','up_proj']}
    embedding_modules = {'embed_tokens':'input_embeddings'}

    def __init__(self, *, vllm_config, prefix=''):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = Qwen3Model(vllm_config=vllm_config,prefix='model')
        self.recipe = self.config.fundm_recipe
        if self.recipe == 'kl':
            self.model.norm = RawFinalState()
        if self.recipe == 'embedding':
            self.pooler = EncoderStatePooler()
        elif self.recipe == 'teacher':
            self.pooler = TeacherLetterPooler(self.config)
        else:
            checkpoint = torch.load(self.config.fundm_head_path,weights_only=False,map_location='cpu')
            if checkpoint['layer'] != self.config.num_hidden_layers:
                raise ValueError('Head and prefix depths disagree')
            self.pooler = DecisionPooler(checkpoint)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self,input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
        return self.model(input_ids,positions,intermediate_tensors,inputs_embeds)

    def load_weights(self,weights):
        def used():
            for name,value in weights:
                match = re.search(r'model\.layers\.(\d+)\.',name)
                if (self.recipe != 'teacher' and name.startswith('lm_head.')) or (self.recipe == 'kl' and name.startswith('model.norm.')) or (match and int(match[1]) >= self.config.num_hidden_layers):
                    continue
                if self.recipe == 'teacher' and name.startswith('lm_head.'):
                    name='pooler.'+name
                yield name,value
        loaded = AutoWeightsLoader(self,skip_prefixes=[] if self.recipe == 'teacher' else ['pooler.']).load_weights(used())
        if self.recipe == 'teacher': return loaded
        return loaded | {name for name,_ in self.named_parameters() if name.startswith('pooler.')}


def memory_probe(worker, reset=False):
    torch.cuda.synchronize()
    if reset:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    kv = getattr(worker.model_runner,'kv_caches',[])
    if isinstance(kv,dict): kv=list(kv.values())
    seen,kv_bytes=set(),0
    for tensor in kv:
        if not isinstance(tensor,torch.Tensor): continue
        storage=tensor.untyped_storage()
        if storage.data_ptr() not in seen:
            seen.add(storage.data_ptr()); kv_bytes+=storage.nbytes()
    return {'cuda_allocated_bytes':torch.cuda.memory_allocated(),
            'cuda_reserved_bytes':torch.cuda.memory_reserved(),
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'kv_tensor_bytes':kv_bytes,
            'model_parameter_bytes':sum(p.numel()*p.element_size() for p in worker.model_runner.model.parameters()),
            'profiled_non_torch_bytes':getattr(worker,'non_torch_memory',None),
            'gpu':torch.cuda.get_device_name()}


class MemoryWorkerExtension:
    def fundm_memory_probe(self, reset=False):
        return memory_probe(self,reset)


ModelRegistry.register_model('FunDMForPooling','fundm_vllm:FunDMForPooling')


class CLMEncoderForPooling(FunDMForPooling):
    default_seq_pooling_type = 'LAST'

ModelRegistry.register_model('CLMEncoderForPooling','fundm_vllm:CLMEncoderForPooling')
