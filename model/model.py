import torch
from torch import nn
import torch.nn.functional as F

# Definition of Entity Resolution Module
class EntityResolution(nn.Module):
    # constraint: span emb dimension same as bag of embedding row dimesion (for dot product)
    # mod: would require dense layer in case dim different
    def __init__(self, triplet_size, max_spans, max_cand, max_prop, num_entities, emb_dim=768):
        super(EntityResolution, self).__init__()
        self.max_spans, self.max_cand, self.max_prop, self.ne = max_spans, max_cand, max_prop, num_entities
        # as attention mask is involed, only sum is supported (refer PyTorch docs)
        self.emb_mean = nn.EmbeddingBag(triplet_size, emb_dim, mode='sum')
        self.softmax = nn.Softmax(dim=1)
        self.span_weights = nn.Linear(emb_dim,1)
        
    def forward(self,
               span_embs,
               triplet_ids_tr,
               offsets_tr,
               attention_tr,
               qid_inds):
        # triplet_ids_tr (b, max_spans*max_cand*max_prop)
        # offsets_tr (b,len(list(range(0,max_spans*max_cand*max_prop))
        # attention_tr (b, max_spans*max_cand*max_prop) (to ask embedding bag to ignore the padded ones)
        # all qid_inds correspnoding to candidates within the sentence (b, max_cands*max_spans)
        # span_embs (b, max_spans, 768)
        # Note: BOE doesnt support 2d inputs, batched not supported.. split batch (bottleneck)
        
        span_scores = self.span_weights(span_embs) 
        exp_emb = span_embs.repeat(1,self.max_cand,1) # repeat max_cand number of times.
        
        # get kb embs
        triplet_ids_splitted = torch.unbind(triplet_ids_tr, dim=0)
        attention_tr_splitted = torch.unbind(attention_tr, dim=0)
        offsets_tr_splitted = torch.unbind(offsets_tr, dim=0)
        
        batch_results = []
        for x,y,z in zip(triplet_ids_splitted, attention_tr_splitted, offsets_tr_splitted):
            batch_results.append(self.emb_mean(x, offsets=z, per_sample_weights=y))
            
        # stack them along to create a new batch dimension
        kb_embs = torch.stack(batch_results, dim=0)
        
            
        mult1 = kb_embs*exp_emb
        mult1 = torch.reshape(mult1, (-1, self.max_spans, self.max_cand, 768))
        sum1 = torch.sum(mult1, dim=3)
        sm1 = self.softmax(sum1)
        
        # multiple span scores
        mult2 = span_scores*sm1
        mult2 = torch.reshape(mult2, (-1, self.max_spans*self.max_cand))
        cand_scores = self.softmax(mult2)
        
        # map to the tensor or zero vector (batched fashion)
        cand_scores = cand_scores.unsqueeze(-1)
        one_hot = F.one_hot(qid_inds, num_classes=self.ne+1).float()
        x0 = torch.matmul(one_hot.permute((0,2,1)),cand_scores)
        # trim the last cell 
        x0 = x0[:,:-1,:]
        # as out of distribution classes are trimmed
        # the remaining vector is passed through softmax again
        # to obtain probability distributions
        # ToDO: reengineer the OOD cases 
        x0 = self.softmax(x0)
        return x0
    

# definition of Inference Module
class InferenceModule(nn.Module):
    def __init__(self, max_hops, Ms, Mo, Mp, emb_dim=768, hdim=80):
        super(InferenceModule, self).__init__()
        self.max_hops = max_hops
        self.softmax = nn.Softmax(dim=1)
        self.weight_inf = nn.ModuleList([nn.Linear(hdim*(hops+1),hdim) for hops in range(0,max_hops)])
        self.weight_attn = nn.ModuleList([nn.Linear(hdim*(hops+1),1) for hops in range(0,max_hops)])
        self.red_dim = nn.Linear(emb_dim, hdim)
        self.Ms, self.Mo, self.Mp = Ms, Mo, Mp

    def forward(self,x,qn_emb):
        # x input from Entity Resolution Module (source entity vector)
        # qn_emb has dim (b, 768)
        qn_emb = self.red_dim(qn_emb)
        r_hist, x_hist, att_hist = [], [], []
        for i in range(self.max_hops):
            r = self.softmax(self.weight_inf[i](torch.cat([qn_emb]+r_hist,dim=1))).unsqueeze(-1)
            x = torch.matmul(torch.transpose(self.Mo,0,1),torch.matmul(self.Ms,x)*torch.matmul(self.Mp,r)) 
            # compute attention scores
            att_score = self.weight_attn[i](torch.cat([qn_emb]+r_hist,dim=1))
            x_hist.append(x.squeeze(-1)) 
            r_hist.append(r.squeeze(-1))
            att_hist.append(att_score)
        
        # concatenate all scores
        attn_scores = self.softmax(torch.cat(att_hist, dim=1)).unsqueeze(-1)
        x_hist = torch.stack(x_hist, dim=1)
        x_final = torch.sum(x_hist*attn_scores, dim=1)
        # x_final: (b, Ne) 
        return x_final
    
class RigelModel(nn.Module):
    
    def __init__(self, 
                 triplet_size, 
                 max_spans, 
                 max_cand, 
                 max_prop, 
                 num_entities, 
                 emb_dim,
                 max_hops, 
                 Ms, 
                 Mo, 
                 Mp,
                 hdim):
        super(RigelModel, self).__init__()
        self.er_model = EntityResolution(
            triplet_size=triplet_size,
            max_spans=max_spans,
            max_cand=max_cand,
            max_prop=max_prop,
            num_entities=num_entities,
            emb_dim=emb_dim
        )

        self.inf_model = InferenceModule(
            max_hops=max_hops,
            Ms=Ms,
            Mo=Mo,
            Mp=Mp,
            hdim=hdim,
            emb_dim=emb_dim
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, 
               span_embs,
               triplet_ids_tr,
               offsets_tr,
               attention_tr,
               qid_inds,
               qn_emb):
        # forward pass through er_model
        out = self.er_model(span_embs, triplet_ids_tr, offsets_tr, attention_tr, qid_inds)
        out = self.inf_model(out, qn_emb)
        # apply sigmoid to final output
        out = self.sigmoid(out)
        return out
    



    

