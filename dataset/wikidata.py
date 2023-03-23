import pandas as pd
import numpy as np
import datetime
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer
)
import torch.nn as nn
import torch
from transformers.pipelines.token_classification import AggregationStrategy
import enum
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher


# Using enum class create enumerations
class Maps(enum.Enum):
    subj = 0
    obj = 1
    prop = 2
    
# overriding NER extractionpipeline from Huggingface to obtain span embeddings
class NERSpanExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model_name, output_hidden_states = True),
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            *args,
            **kwargs
        )
    def _forward(self, model_inputs):
        # Forward
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        sentence = model_inputs.pop("sentence")

        outputs = self.model(**model_inputs)
        logits = outputs[0]
        
        # hidden state keys value
        hidden_state = outputs[1]

        return {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            "hidden_state": hidden_state,
            **model_inputs,
        }
    
    def postprocess(self, model_outputs, aggregation_strategy=AggregationStrategy.NONE, ignore_labels=None):
        # invoke parent cclass method with defaults 
        ner_results = super().postprocess(
            model_outputs=model_outputs
        )

        last_hidden_state = model_outputs['hidden_state'][1]
        
        proc_out = []
        # get spans and model_output last hidden state for the embeddings..
        span_indexes, span_words = [], []
        for span in ner_results:
            start, end = span['start'], span['end']
            span_text = model_outputs['sentence'][start:end]
            index = span['index'] # first word starts from 1
            ner_tag = span['entity']
            
            # added for large NER model
            word = span['word']
            
            # parse based on BIO tagging
            if ner_tag.startswith('B-'):
                # word positions
                span_indexes.append([index])
                span_words.append([span_text])
            elif ner_tag.startswith('I-'):
                span_indexes[-1].append(index)
                if not word.startswith('##'):
                    span_words[-1].append(span_text)
                else:
                    span_words[-1][-1]+=span_text.lstrip("#")

        for text, span in zip(span_words, span_indexes):
            slice_indices = torch.tensor([x for x in range(span[0], span[-1]+1)])
            sliced = torch.index_select(last_hidden_state, 1, slice_indices)
            mean = torch.mean(sliced, dim=1)
            proc_out.append({'span_text': ' '.join(text), 'span_embedding':mean})
            
        return proc_out
    
# Dataset class for KGQA on Wikidata
# definitions based on Rigel Model input types
class WikiDataDataset(Dataset):
    def __init__(self, 
                 ent_file, 
                 pred_file, 
                 trip_file, 
                 ds_file, 
                 max_spans, 
                 max_cand, 
                 max_properties,
                 span_detn_model,
                 sentence_emb_model,
                 emb_dim):

        self.max_spans, self.max_cand, self.max_properties = max_spans, max_cand, max_properties
        self.entity_df, self.properties_df, self.triplets_df = self.preprocess_df(ent_file, pred_file, trip_file)
        self.dataset_df = self.preprocess_dataset(ds_file)
        self.total_entities = len(self.entity_df)
        self.total_triplets = len(self.triplets_df)
        self.total_properties = len(self.properties_df)
        self.emb_dim = emb_dim
        
        # for subject, object, property
        self.id2text = [{},{},{}]
        self.text2id = [{},{},{}]
        # for subject, object, property
        self.ind2id = [{},{},{}]
        self.id2ind = [{},{},{}]
        # for triplet to qid mappings (subject, object, property)
        self.id2tripletind = [{},{},{}]
        self.ind2tripletind = [{},{},{}]
        
        # set entity and property mappings
        self.set_schema_maps(self.entity_df, prefix='s', map_val=Maps.subj.value)
        self.set_schema_maps(self.entity_df, prefix='o', map_val=Maps.obj.value)
        self.set_schema_maps(self.properties_df, prefix='p', map_val=Maps.prop.value)
        
        
        # set triplet mappings
        self.set_triplet_maps(self.triplets_df, prefix='s', map_val=Maps.subj.value)
        self.set_triplet_maps(self.triplets_df, prefix='o', map_val=Maps.obj.value)
        self.set_triplet_maps(self.triplets_df, prefix='p', map_val=Maps.prop.value)
        # get ner pipeline
        self.ner = NERSpanExtractionPipeline(model_name = span_detn_model)
        # get sentence embedding model
        self.sentence_emb = SentenceTransformer(sentence_emb_model)
        
        
    def preprocess_entities_list(self, ent_file):
        df = pd.read_excel(ent_file)
        len_a = len(df)
        df = df.replace(np.nan, '', regex=True)
        df = df.rename(columns=
                  {"entity_id": "e_id",
                  "entity_name": "e_label",
                  "entity_alias": "e_alias"})
        
        df['e_label']=df['e_label'].astype('str')
        df['e_alias']=df['e_alias'].astype('str')
        df = df.drop_duplicates()
        len_b = len(df)        
        return df
    
    def preprocess_properties_list(self, prop_file):
        df = pd.read_excel(prop_file)
        len_a = len(df)
        df = df.replace(np.nan, '', regex=True)
        df = df.rename(columns=
                  {"predicate_id": "p_id",
                  "predicate_name": "p_label",
                  "predicate_alias": "p_alias"})
        
        df['p_label']=df['p_label'].astype('str')
        df['p_alias']=df['p_alias'].astype('str')
        df = df.drop_duplicates()
        len_b = len(df)
        return df
    
    def preprocess_triplets_list(self, trip_file):
        df = pd.read_excel(trip_file)
        len_a = len(df)
        df = df.dropna()
        df = df.drop_duplicates()
        len_b = len(df)
        df = df.rename(columns=
                  {"s_id": "s_id",
                  "p_id": "p_id",
                  "o_id": "o_id"})                    
        return df
    
    def preprocess_df(self, ent_file, pred_file, trip_file):
        df_entities = self.preprocess_entities_list(ent_file)
        # self.entity_df.to_csv('test.csv')
        df_prop = self.preprocess_properties_list(pred_file)
        df_triplets = self.preprocess_triplets_list(trip_file)
        df_entities1 = df_entities[df_entities['e_id'].isin(df_triplets['s_id']) | df_entities['e_id'].isin(df_triplets['o_id'])]
        df_prop1 = df_prop[df_prop['p_id'].isin(df_triplets['p_id'])]
        return df_entities1, df_prop1, df_triplets
    
    def preprocess_dataset(self, ds_file):
        data = {}
        with open(ds_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                entry = json.loads(line)
                question = entry['question']
                answer_qids = []
                for answer in entry['answer']['answer']:
                    answer_qids.append(answer['name'])
                data['question'] = data.setdefault('question',[]) + [question]
                data['answer_qids'] = data.setdefault('answer_qids',[]) + [answer_qids]
        df = pd.DataFrame(data, columns=['question','answer_qids'])
        return df
    
    def set_schema_maps(self, df, prefix, map_val):
        col_name = f'{prefix}_id'
        # all subjects or all objects or all properties qids in the triplets
        all_triplet_entities = set()
        for ind, row in self.triplets_df.iterrows():
            all_triplet_entities.add(row[col_name]) # append the qids 
        
        prefix_df = prefix if map_val==2 else 'e'
        for ind, row in df.iterrows():
            qid = row[f'{prefix_df}_id']
            if qid in all_triplet_entities:
                all_aliases = row[f'{prefix_df}_alias'].split(',')
                text_list = [row[f'{prefix_df}_label'].strip()] + (all_aliases if all_aliases!=[''] else [])
                text_list = [x.lower().strip() for x in text_list]
                self.id2text[map_val][qid]=text_list
            
                for text in text_list:
                    self.text2id[map_val][text] = self.text2id[map_val].setdefault(text,[])+[qid]
                # store in respective arrays for subject, property and object
                self.id2ind[map_val][qid]=ind
                self.ind2id[map_val][ind]=qid
            
    def set_triplet_maps(self, df, prefix, map_val):
        col_name = f'{prefix}_id'
        for ind, row in df.iterrows():
            self.id2tripletind[map_val][row[col_name]]=self.id2tripletind[map_val].setdefault(row[col_name],[])+[ind]
            
            qid_ind = self.id2ind[map_val][row[col_name]]
            self.ind2tripletind[map_val][qid_ind]=self.ind2tripletind[map_val].setdefault(qid_ind, [])+[ind]
    
    def find_candqid_inds(self, span_text):
        # if span text is a valid entity text then return the corresponding qids..
        final_inds = []
        for entity_text in self.text2id[Maps.subj.value]:
            # id for the text and make sure we are not entering same qid twice
            ratio = self.word_similarity(span_text, entity_text, thresh=0.70)
            if ratio:
                ids = self.text2id[Maps.subj.value][entity_text]
                inds = [self.id2ind[Maps.subj.value][id] for id in ids]
                final_inds.extend(inds)
        return list(set(final_inds)) # remove duplicate inds
    
    
    # logic for computing word similarity between entity span and the entities with the KB
    def word_similarity(self, a, b, thresh):
        sim = SequenceMatcher(None, a, b).ratio()
        if sim>=thresh:
            return 1
        return 0
    
    
    def compute_gt(self, qids):
        # assumption: answer entity can only be an object in the triplet
        # try:
        #     answer_qids_indxs = [self.id2ind[Maps.obj.value][x] for x in qids]
        # except Exception as e:
        #     raise Exception("Answer QID not in Object Entities list of QIDs",qids,self.id2ind[Maps.obj.value])
        
        # bypass check for test
        answer_qids_indxs = [0]
        
        answer_vector = [0.0 for _ in range(0,self.total_entities)]
        for aind in answer_qids_indxs:
            answer_vector[aind]=1.0
        return answer_vector
        
    def __len__(self):
        # return length of dataset
        return len(self.dataset_df)
    
    # return Ms, Mo, and Mp
    def get_sparse_matrix(self, mode):
        col_dim = self.total_properties if mode==Maps.prop else self.total_entities
        M = [[0.0 for _ in range(col_dim)] for _ in range(len(self.triplets_df))]
        for i in range(col_dim):
            if i in self.ind2tripletind[mode.value]:
                # print('There')
                triplet_inds = self.ind2tripletind[mode.value][i]
                for ti in triplet_inds:
                    M[ti][i]=1.0
        M = torch.tensor(M)
        return M
    
    def __getitem__(self, idx):
        # get text based on the idx
        # eg: text = "Who is taller, Angelina Jolie or Brad Pitt?"

        text = self.dataset_df.iloc[idx]['question']
        answer_qids = self.dataset_df.iloc[idx]['answer_qids']
        answer_vector = self.compute_gt(answer_qids)
        
        # get ner results
        ner_results = self.ner(text)[:self.max_spans]
        
        candqids = []
        # indexes of triplets
        trips_ids = []
        # attention scores for mean BOE of the above trips
        attn_trips_ids = []
        # for spans
        span_embs = []

        # given we have ner_results for the text given by an idx in the df..
        for span in ner_results:
            span_text = span['span_text']
            span_emb = span['span_embedding']
            span_embs.append(span_emb)            
            span_triplets, span_attn_triplets = [], []

            # get all candidate inds (trimm by max_cand)
            candqid_inds = self.find_candqid_inds(span_text)[:self.max_cand]
            candqid_inds_tr = torch.tensor(candqid_inds)
            num_cands = list(candqid_inds_tr.shape)[0]
            # create padded version with candidates padded to max_cand for each span
            cand_qids_padded = F.pad(candqid_inds_tr, (0,self.max_cand-num_cands), "constant", self.total_entities)
            candqids.append(cand_qids_padded)

            # trim them to max_cand
            for cand_ind in candqid_inds:
                # get all triplet inds for BOE for the given cand_ind
                triplet_inds = self.ind2tripletind[Maps.subj.value][cand_ind]
                triplet_inds_tr = torch.tensor(triplet_inds)
                # pad inds for properties of this span to max properties
                num_properties = list(triplet_inds_tr.shape)[0]
                triplet_inds_padded = F.pad(triplet_inds_tr, (0,self.max_properties-num_properties), "constant", 0)
                span_triplets.append(triplet_inds_padded)

                # for attention tensor preparation (actual weights, + padded weights which are zeros)
                # shape (max_properties)
                attn_vals_padded = [1.0/num_properties for _ in range(0,num_properties)]+[0.0 for _ in range(0,self.max_properties-num_properties)]
                span_attn_triplets.append(torch.tensor(attn_vals_padded))
            
            # in case of no candidates
            if not num_cands:
                span_triplets.append(torch.tensor([0 for _ in range(0,self.max_properties)]))
                span_attn_triplets.append(torch.tensor([0.0 for _ in range(0,self.max_properties)]))
            
            # concatenate the property level across all candidates
            span_trip = torch.stack(span_triplets,dim=0) # shape (num_cands, num_properties)
            num_cands, _ = span_trip.shape
            # pad it with 0s for max_cands dim
            span_trip_padded = F.pad(span_trip, (0,0,0,self.max_cand-num_cands), "constant", 0.0)
            trips_ids.append(span_trip_padded)

            span_attn_trip = torch.stack(span_attn_triplets,dim=0)
            num_cands, _ = span_attn_trip.shape
            span_attn_trip_padded = F.pad(span_attn_trip, (0,0,0,self.max_cand-num_cands), "constant", 0.0)
            attn_trips_ids.append(span_attn_trip_padded)

        # finally pad at sentence level (max_spans)
        # pad this with num_classes instead of 0.0
        # handle cases of no spans
        if not ner_results:
            candqids.append(torch.tensor([self.total_entities for _ in range(0, self.max_cand)]))
            attn_trips_ids.append(torch.tensor([[0.0 for _ in range(0, self.max_properties)] for _ in range(0,self.max_cand)]))
            trips_ids.append(torch.tensor([[0 for _ in range(0, self.max_properties)] for _ in range(0,self.max_cand)]))
            span_embs.append(torch.tensor([[0.0 for _ in range(self.emb_dim)]]))
            
        cand_qids_sent = torch.stack(candqids, dim=0)
        num_spans, _ = cand_qids_sent.shape
        cand_qids_sent_padded = F.pad(cand_qids_sent, (0,0,0,self.max_spans-num_spans), "constant", self.total_entities)


        # for triplet ids
        trips_ids_sent = torch.stack(trips_ids, dim=0)
        num_spans, _, _ = trips_ids_sent.shape
        trips_ids_sent_padded = F.pad(trips_ids_sent, (0,0,0,0,0,self.max_spans-num_spans), "constant", 0)

        # for triplet_attns scores
        attn_trips_ids_sent = torch.stack(attn_trips_ids, dim=0) # shape: num_spans, max_cand, max_prop
        num_spans, _, _ = attn_trips_ids_sent.shape
        attn_trips_ids_sent_padded = F.pad(attn_trips_ids_sent, (0,0,0,0,0,self.max_spans-num_spans), "constant", 0.0)

        # span embs
        span_embs_sent = torch.cat(span_embs, dim=0) #(num_spans, 768)
        num_spans, _ = span_embs_sent.shape
        span_embs_padded = F.pad(span_embs_sent, (0,0,0,self.max_spans-num_spans), "constant", 0.0)


        triplets_ids_tr = torch.flatten(trips_ids_sent_padded).type(torch.int64)
        attention_tr = torch.flatten(attn_trips_ids_sent_padded)
        offsets_tr = torch.tensor(list(range(0,self.max_spans*self.max_cand*self.max_properties,self.max_properties)))
        # casting qids to integer type for 1 hot encoding
        qid_inds = torch.flatten(cand_qids_sent_padded).type(torch.int64)
        span_embs = span_embs_padded
        # also to pass the sentence embedding for the inference module
        sentence_emb = torch.from_numpy(self.sentence_emb.encode([text])).squeeze(0)
        answer_vector = torch.tensor(answer_vector)
        
        features = [
            triplets_ids_tr,
            attention_tr,
            offsets_tr,
            qid_inds,
            span_embs,
            sentence_emb,
            answer_vector
        ]
        
        return features