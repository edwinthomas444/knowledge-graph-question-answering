import sys
sys.path.append('./')

from dataset.wikidata import WikiDataDataset
from torch.utils.data import DataLoader
from model.model import RigelModel
from dataset.wikidata import Maps
import torch.nn as nn

def main():
    # Define Dataset class
    ent_file = '../KB_v3/entity_csv.xlsx'
    prop_file = '../KB_v3/predicate_csv.xlsx'
    trip_file = '../KB_v3/triplet_csv.xlsx'
    ds_file = '../Dataset_v1/dev_30.txt'
    span_model = 'Davlan/bert-base-multilingual-cased-ner-hrl'
    sentence_model = 'sentence-transformers/all-distilroberta-v1'
    
    # note: hidden emb dim must be same for both sentence and span detn models (i.e 768)
    dataset = WikiDataDataset(  
                            ent_file,
                            prop_file,
                            trip_file,
                            ds_file,
                            max_cand=3,
                            max_spans=12,
                            max_properties=10,
                            span_detn_model=span_model,
                            sentence_emb_model=sentence_model,
                            emb_dim=768)
    
    # obtain the train dataloader
    train_dataloader = DataLoader(dataset, batch_size=2)

    model = RigelModel(
        triplet_size=dataset.total_triplets,
        max_spans=12,
        max_cand=3,
        max_prop=10,
        num_entities=dataset.total_entities,
        max_hops=1,
        Ms=dataset.get_sparse_matrix(Maps.subj).to('cuda'),
        Mo=dataset.get_sparse_matrix(Maps.obj).to('cuda'),
        Mp=dataset.get_sparse_matrix(Maps.prop).to('cuda'),
        hdim=dataset.total_properties,
        emb_dim=768
    )
    model.to('cuda')

    # define loss
    loss = nn.BCELoss()

    # define total epochs
    total_epochs = 1

    # train for total number of epochs
    for epoch in range(0,total_epochs):
        for batch in train_dataloader:
            # move the batch to gpu
            batch = tuple(t.to('cuda') for t in batch)
            inputs_er = {
                "span_embs":batch[4],
                "triplet_ids_tr":batch[0],
                "offsets_tr":batch[2],
                "attention_tr":batch[1],
                "qid_inds":batch[3],
                "qn_emb":batch[5]
            }

            model.train()

            out = model(**inputs_er)
            out_loss = loss(out, batch[6])

            print(f'Epoch: {epoch}, Loss: {out_loss.item()}')
            out_loss.backward()



    
if __name__ == '__main__':
    main()