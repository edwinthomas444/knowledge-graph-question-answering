import sys
sys.path.append('./')

from dataset.wikidata import WikiDataDataset
from torch.utils.data import DataLoader
from model.model import RigelModel
from dataset.wikidata import Maps
import torch.nn as nn
from tqdm import tqdm 
import torch
import json
import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils.scoring import multi_label_metrics

def config_parser(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    '''
    example usage:
    python ./train.py --config './configs/base.json'
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    configs = config_parser(args.config)


    # Define Dataset class
    # note: hidden emb dim must be same for both sentence and span detn models (i.e 768)
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
    
    # obtain the train dataloader
    train_dataloader = DataLoader(dataset, batch_size=configs['train']['batch_size'])

    model = RigelModel(
        triplet_size=dataset.total_triplets,
        max_spans=configs['hparams']['max_spans'],
        max_cand=configs['hparams']['max_cand'],
        max_prop=configs['hparams']['max_properties'],
        num_entities=dataset.total_entities,
        max_hops=configs['hparams']['max_hops'],
        Ms=dataset.get_sparse_matrix(Maps.subj).to('cuda'),
        Mo=dataset.get_sparse_matrix(Maps.obj).to('cuda'),
        Mp=dataset.get_sparse_matrix(Maps.prop).to('cuda'),
        hdim=dataset.total_properties,
        emb_dim=configs['hparams']['emb_size']
    )
    print(model)
    model.to('cuda')

    

    # define loss
    loss = nn.BCELoss()

    # define total epochs
    total_epochs = configs['train']['epochs']

    # define lr schedulers and optimizers
    optimizer = optim.Adam(model.parameters(), lr=configs['train']['lr'])
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    
    # train for total number of epochs
    for epoch in range(0,total_epochs):
        ep_preds, ep_gt = [], []

        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
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
            # add batch preds and gt

            # remove copy of output tensors from computation graph
            ep_preds.append(out.detach().to('cpu'))
            ep_gt.append(batch[6].detach().to('cpu'))   

            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()

            # save the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': out_loss.item(),
                }, configs['train']['save_path'])
        
        # compute scores
        result = multi_label_metrics(
                                    y_pred=torch.cat(ep_preds, dim=0).tolist(),
                                    y_true=torch.cat(ep_gt, dim=0).tolist(),
                                    thresh=0.5,
                                    labels=dataset.entity_labels)
        # save metrics for each epoch
        result.to_csv(f'./results/result1_{epoch}.csv')

        # step lr after each epoch
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print(f'Epoch: {epoch}, Loss: {out_loss.item()}, LR: {after_lr}')

if __name__ == '__main__':
    main()