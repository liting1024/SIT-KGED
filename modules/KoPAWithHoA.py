import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple

from transformers import LlamaForCausalLM
import numpy as np


class KoPAWithHoA(nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        num_prefix: int,
        ent_embeddings,
        rel_embeddings
    ) -> None:
        super(KoPAWithHoA, self).__init__()
        self.llama_model = model

        self.embeddings = HoA(
            hoa_list=ent_embeddings,
            rel_embeddings=rel_embeddings,
            dim_llm=model.model.lm_head.in_features,
            num_prefix=num_prefix
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
    

class HoA(nn.Module):
    def __init__(
        self,
        hoa_list,
        rel_embeddings,
        dim_llm,
        num_prefix,
        hop=3,
        dropout=0.1,
    ):
        super(HoA, self).__init__()
        self.num_prefix = num_prefix
        self.llm_dim = dim_llm  
        self.emb_dim = num_prefix * dim_llm
        self.hop = hop

        self.rel_embeddings = rel_embeddings
        self.hoa_list = hoa_list

        self.pretrain_dim = self.rel_embeddings.shape[1]
        self.adapter = nn.Linear(self.pretrain_dim, self.emb_dim)
        

        self.att = nn.Parameter(torch.ones(hop))
        self.sm = nn.Softmax(dim=0)
        self.linear_layers = nn.ModuleList([MLP(num_layers=1, in_dim=self.pretrain_dim, out_dim=self.pretrain_dim, activation=nn.ReLU) for _ in range(hop)])
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.pretrain_dim*hop, self.pretrain_dim)
        )
    
    def get_hoa(self):
        '''  '''
        mask = self.sm(self.att)
        linear_features = []
        for i in range(self.hop):
            linear_out = self.linear_layers[i](self.hoa_list[i])
            linear_features.append(torch.mul(mask[i], linear_out))

        concat_out = torch.cat(linear_features, dim=1)
        out = self.fc(concat_out)

        # out = F.log_softmax(out)  # dont need
        return out
    
    def forward(self, triple_ids):
        self.ent_embeddings = self.get_hoa()

        head, relation, tail = triple_ids[:, 0], triple_ids[:, 1], triple_ids[:, 2]
        h = self.ent_embeddings[head]
        r = self.rel_embeddings[relation]
        t = self.ent_embeddings[tail]

        pretrain_embs = torch.stack((h, r, t), dim=1)
        prefix = self.adapter(pretrain_embs).reshape(-1, 3*self.num_prefix, self.llm_dim)

        return prefix


class MLP(nn.Module):
    def __init__(self, num_layers, in_dim, out_dim, activation=nn.ReLU):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        if isinstance(in_dim, int):
            in_dim = [in_dim]*num_layers
        if isinstance(out_dim, int):
            out_dim = [out_dim]*num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_dim[i], out_dim[i]))
            if i < num_layers - 1:
                self.layers.append(activation())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

