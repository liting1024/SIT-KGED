import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple

from transformers import LlamaForCausalLM
from pykeen.datasets import get_dataset, PathDataset
from torch_sparse import SparseTensor
from preprocess.node_label import HoCN

class SIT(nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        num_prefix: int,
        ent_embeddings,
        rel_embeddings,
        dataset=None,
    ) -> None:
        super(SIT, self).__init__()
        self.llama_model = model

        self.embeddings = InjectTopo(
            ent_embeddings=ent_embeddings,
            rel_embeddings=rel_embeddings,
            dim_llm=model.model.lm_head.in_features,
            num_prefix=num_prefix,
            dataset=dataset,
        )

    
    def forward(
        self,
        input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        embedding_ids: torch.LongTensor = None
    ):
        kg_embeds = self.embeddings(embedding_ids)
        # print(kg_embeds.shape)
        batch_size, seq_len, _ = kg_embeds.shape
        token_embeds = self.llama_model.model.model.embed_tokens(input_ids)
        input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)
        prefix_mask = torch.ones((batch_size, seq_len))
        prefix_labels = torch.full((batch_size, seq_len), fill_value=-100, dtype=torch.long)
        new_attention_mask = torch.cat((prefix_mask.cuda(), attention_mask), dim=-1)
        new_labels = torch.cat((prefix_labels.cuda(), labels), dim=-1)
        return self.llama_model(
            input_ids=None,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
class InjectTopo(nn.Module):
    def __init__(
        self,
        ent_embeddings,
        rel_embeddings,
        dim_llm,
        num_prefix,
        dataset
    ):
        super(InjectTopo, self).__init__()
        self.num_prefix = num_prefix
        self.llm_dim = dim_llm
        self.emb_dim = num_prefix * dim_llm
        self.ent_embeddings = ent_embeddings
        self.rel_embeddings = rel_embeddings
        self.pretrain_dim = self.ent_embeddings.shape[1]
        # Froze the pretrain embeddings

        self.adapter = nn.Linear(self.pretrain_dim, self.emb_dim)
        self.moe_adapter = MoEAdaptorLayer(
            n_exps=7,
            layers=[self.pretrain_dim, self.emb_dim],
            dropout=0.1,
            noise=True
        )
        
        
        dataset = get_dataset(dataset=dataset)

        dataset = dataset.training
        triples = dataset.mapped_triples
        row = triples[:, 0]  # subject
        col = triples[:, 2]  # object
        rel = torch.ones_like(row) 
        self.adj_t = SparseTensor(row=row, col=col, value=rel, sparse_sizes=(dataset.num_entities, dataset.num_entities)).to_device(self.ent_embeddings.device)

        self.node_label = HoCN()


    def forward(self, triple_ids):
        # main training stage
        head, relation, tail = triple_ids[:, 0], triple_ids[:, 1], triple_ids[:, 2] 
        # TODO move to preprocessing
        count_1_1, count_1_2, count_2_1, count_2_2,  = self.node_label.forward(
            x=self.ent_embeddings,
            edges=[head, tail],
            adj_t=self.adj_t,
        )
        sub_features_list = [
            self.ent_embeddings[head], 
            self.rel_embeddings[relation], 
            self.ent_embeddings[tail],
            count_1_1,
            count_1_2,
            count_2_1,
            count_2_2,
        ]

        sub_features = torch.stack(sub_features_list, dim=1)
        sub_features = sub_features.to(dtype=torch.float16)

        prefix = self.moe_adapter(sub_features).reshape(-1, len(sub_features_list)*self.num_prefix, self.llm_dim)
        return prefix
    



class PWLayer(nn.Module):
    """ Single Parametric Whitening Layer """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)
    

class MoEAdaptorLayer(nn.Module):
    """ MoE-enhanced Adaptor """
    def __init__(self, n_exps=4, layers=[512, 4096], dropout=0.2, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, hoa_list):
        gates = self.noisy_top_k_gating(hoa_list, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](hoa_list).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)