import sys
sys.path.append('./')

from model.model import RigelModel
from dataset.wikidata import WikiDataDataset
from model.model import RigelModel
from dataset.wikidata import Maps
import torch
import torch.nn.functional as F
import json
import argparse



def config_parser(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

class Inference:
    def __init__(self, configs):
        # Define Dataset class
        self.configs = configs
        self.dataset = self.get_dataset(configs)
        
    def get_dataset(self, configs):
        dataset = WikiDataDataset(  
                            configs['dataset']['ent_file'],
                            configs['dataset']['prop_file'],
                            configs['dataset']['trip_file'],
                            configs['dataset']['ds_file'],
                            max_cand=configs['hparams']['max_cand'],
                            max_spans=configs['hparams']['max_spans'],
                            max_properties=configs['hparams']['max_properties'],
                            span_detn_model=configs['hparams']['span_model'],
                            sentence_emb_model=configs['hparams']['sentence_model'],
                            emb_dim=configs['hparams']['emb_size'])
        return dataset

    def get_model(self, checkpoint=''):
        # define the model
        model = RigelModel(
            triplet_size=self.dataset.total_triplets,
            max_spans=self.configs['hparams']['max_spans'],
            max_cand=self.configs['hparams']['max_cand'],
            max_prop=self.configs['hparams']['max_properties'],
            num_entities=self.dataset.total_entities,
            max_hops=self.configs['hparams']['max_hops'],
            Ms=self.dataset.get_sparse_matrix(Maps.subj).to('cuda'),
            Mo=self.dataset.get_sparse_matrix(Maps.obj).to('cuda'),
            Mp=self.dataset.get_sparse_matrix(Maps.prop).to('cuda'),
            hdim=self.dataset.total_properties,
            emb_dim=self.configs['hparams']['emb_size']
        )
        checkpt = torch.load(checkpoint)
        model.load_state_dict(checkpt['model_state_dict'])
        return model
    
    def get_features(self, question):
        # get ner results
        ner_results = self.dataset.ner(question)[:self.dataset.max_spans]

        candqids, trips_ids, attn_trips_ids, span_embs = [], [], [], []

        # given we have ner_results for the text given by an idx in the df..
        for span in ner_results:
            span_text = span['span_text']
            span_emb = span['span_embedding']
            span_embs.append(span_emb)            
            span_triplets, span_attn_triplets = [], []

            # get all candidate inds (trimm by max_cand)
            candqid_inds = self.dataset.find_candqid_inds(span_text)[:self.dataset.max_cand]
            candqid_inds_tr = torch.tensor(candqid_inds)
            num_cands = list(candqid_inds_tr.shape)[0]
            # create padded version with candidates padded to max_cand for each span
            cand_qids_padded = F.pad(candqid_inds_tr, (0,self.dataset.max_cand-num_cands), "constant", self.dataset.total_entities)
            candqids.append(cand_qids_padded)

            # trim them to max_cand
            for cand_ind in candqid_inds:
                # get all triplet inds for BOE for the given cand_ind
                # trip properties to max length
                triplet_inds = self.dataset.ind2tripletind[Maps.subj.value][cand_ind][:self.dataset.max_properties]
                triplet_inds_tr = torch.tensor(triplet_inds)
                # pad inds for properties of this span to max properties
                num_properties = list(triplet_inds_tr.shape)[0]
                triplet_inds_padded = F.pad(triplet_inds_tr, (0,self.dataset.max_properties-num_properties), "constant", 0)
                span_triplets.append(triplet_inds_padded)

                # for attention tensor preparation (actual weights, + padded weights which are zeros)
                # shape (max_properties)
                attn_vals_padded = [1.0/num_properties for _ in range(0,num_properties)]+[0.0 for _ in range(0,self.dataset.max_properties-num_properties)]
                span_attn_triplets.append(torch.tensor(attn_vals_padded))
            
            # in case of no candidates
            if not num_cands:
                span_triplets.append(torch.tensor([0 for _ in range(0,self.dataset.max_properties)]))
                span_attn_triplets.append(torch.tensor([0.0 for _ in range(0,self.dataset.max_properties)]))
            
            # concatenate the property level across all candidates
            span_trip = torch.stack(span_triplets,dim=0) # shape (num_cands, num_properties)
            num_cands, _ = span_trip.shape
            # pad it with 0s for max_cands dim
            span_trip_padded = F.pad(span_trip, (0,0,0,self.dataset.max_cand-num_cands), "constant", 0.0)
            trips_ids.append(span_trip_padded)

            span_attn_trip = torch.stack(span_attn_triplets,dim=0)
            num_cands, _ = span_attn_trip.shape
            span_attn_trip_padded = F.pad(span_attn_trip, (0,0,0,self.dataset.max_cand-num_cands), "constant", 0.0)
            attn_trips_ids.append(span_attn_trip_padded)

        # finally pad at sentence level (max_spans)
        # pad this with num_classes instead of 0.0
        # handle cases of no spans
        if not ner_results:
            candqids.append(torch.tensor([self.dataset.total_entities for _ in range(0, self.dataset.max_cand)]))
            attn_trips_ids.append(torch.tensor([[0.0 for _ in range(0, self.dataset.max_properties)] for _ in range(0,self.dataset.max_cand)]))
            trips_ids.append(torch.tensor([[0 for _ in range(0, self.dataset.max_properties)] for _ in range(0,self.dataset.max_cand)]))
            span_embs.append(torch.tensor([[0.0 for _ in range(self.dataset.emb_dim)]]))
            
        cand_qids_sent = torch.stack(candqids, dim=0)
        num_spans, _ = cand_qids_sent.shape
        cand_qids_sent_padded = F.pad(cand_qids_sent, (0,0,0,self.dataset.max_spans-num_spans), "constant", self.dataset.total_entities)


        # for triplet ids
        trips_ids_sent = torch.stack(trips_ids, dim=0)
        num_spans, _, _ = trips_ids_sent.shape
        trips_ids_sent_padded = F.pad(trips_ids_sent, (0,0,0,0,0,self.dataset.max_spans-num_spans), "constant", 0)

        # for triplet_attns scores
        attn_trips_ids_sent = torch.stack(attn_trips_ids, dim=0) # shape: num_spans, max_cand, max_prop
        num_spans, _, _ = attn_trips_ids_sent.shape
        attn_trips_ids_sent_padded = F.pad(attn_trips_ids_sent, (0,0,0,0,0,self.dataset.max_spans-num_spans), "constant", 0.0)

        # span embs
        span_embs_sent = torch.cat(span_embs, dim=0) #(num_spans, 768)
        num_spans, _ = span_embs_sent.shape
        span_embs_padded = F.pad(span_embs_sent, (0,0,0,self.dataset.max_spans-num_spans), "constant", 0.0)


        triplets_ids_tr = torch.flatten(trips_ids_sent_padded).type(torch.int64)
        attention_tr = torch.flatten(attn_trips_ids_sent_padded)
        offsets_tr = torch.tensor(list(range(0,self.dataset.max_spans*self.dataset.max_cand*self.dataset.max_properties,self.dataset.max_properties)))
        # casting qids to integer type for 1 hot encoding
        qid_inds = torch.flatten(cand_qids_sent_padded).type(torch.int64)
        span_embs = span_embs_padded
        # also to pass the sentence embedding for the inference module
        sentence_emb = torch.from_numpy(self.dataset.sentence_emb.encode([question])).squeeze(0)
        
        # add batch dimension
        features = [
            triplets_ids_tr.unsqueeze(0),
            attention_tr.unsqueeze(0),
            offsets_tr.unsqueeze(0),
            qid_inds.unsqueeze(0),
            span_embs.unsqueeze(0),
            sentence_emb.unsqueeze(0)
        ]
        return features

    # infer 1 sentence at a time
    def infer(self, question):
        # get input features
        features = self.get_features(question)

        # get model
        model = self.get_model(checkpoint=self.configs['inference']['rigel_model'])
        model.to('cuda')
        model.eval()

        # forward pass through model
        with torch.no_grad():
            batch = tuple(t.to('cuda') for t in features)

            test_features = {
                "span_embs":batch[4],
                "triplet_ids_tr":batch[0],
                "offsets_tr":batch[2],
                "attention_tr":batch[1],
                "qid_inds":batch[3],
                "qn_emb":batch[5]
            }

            out = model(**test_features)

            # get answer candidates indices
            # its possible that all the weightage can go to ood class when the initial vector
            # is passed for padded entities.
            # Note: output is multilabel
            if self.configs['inference']['multilabel']:
                ans_qid_inds = torch.where(out>=self.configs['inference']['thresh'])[1].to('cpu').tolist()
            else:
                ans_qid_inds = [torch.argmax(out.to('cpu')).item()]
            answers = [self.dataset.entity_df.iloc[ind]['e_label'] for ind in ans_qid_inds]
            print(answers)
            result = {
                'Question':question,
                'Answers':answers
            }
            return result


def main():
    '''
    Sample usage: 
    python .\inference.py --config './configs/base.json'
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = config_parser(args.config)

    # inference pipeline
    inf_handle = Inference(configs=config)
    result = inf_handle.infer(config['inference']['question'])

    # display results
    print('Question: ',result['Question'])
    print('Answers: ', result['Answers'])

if __name__ == "__main__":
    main()