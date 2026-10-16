import torch

import os
import time
import h5py
import argparse
from torch.utils.data import DataLoader
from data_provider.data_loader_save import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from gen_prompt_emb import GenPromptEmb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--input_len", type=int, default=96)
    parser.add_argument("--output_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--l_layers", type=int, default=12)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--divide", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=min(10, os.cpu_count()))
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    return parser.parse_args()

def get_dataset(data_path, flag, input_len, output_len, target):
    datasets = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    dataset_class = datasets.get(data_path, Dataset_Custom)
    return dataset_class(flag=flag, size=[input_len, 0, output_len], data_path=data_path, target=target)

def save_embeddings(args):
    if torch.cuda.is_available():
        device = torch.device(args.device)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    train_set = get_dataset(args.data_path, 'train', args.input_len, args.output_len, args.target)
    test_set = get_dataset(args.data_path, 'test', args.input_len, args.output_len, args.target)
    val_set = get_dataset(args.data_path, 'val', args.input_len, args.output_len, args.target)

    data_loader = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        'test': DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        'val': DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    }[args.divide]

    gen_prompt_emb = GenPromptEmb(
        device=device, # type: ignore
        input_len=args.input_len,
        data_path=args.data_path,
        model_name=args.model_name,
        d_model=args.d_model,
        layer=args.l_layers,
        divide=args.divide,
        embedding_batch_size=args.embedding_batch_size
    ).to(device)

    save_path = f"./Embeddings/{args.data_path}/s{args.input_len}_h{args.output_len}/{args.divide}/"
    os.makedirs(save_path, exist_ok=True)

    emb_time_path = f"./Results/emb_logs/"
    os.makedirs(emb_time_path, exist_ok=True)

    for batch_index, (x, y, x_mark, y_mark) in enumerate(data_loader):
        embeddings = gen_prompt_emb.generate_embeddings(
            x.to(device, dtype=torch.float32), x_mark.to(device, dtype=torch.float32)
        )
        batch_count = x.shape[0]
        for row in range(batch_count):
            sample_index = batch_index * args.batch_size + row
            file_path = f"{save_path}{sample_index}.h5"
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('embeddings', data=embeddings[row:row + 1].cpu().numpy())

        # # Save and visualize the first sample
        # if i >= 0:
        #     break
    
if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    save_embeddings(args)
    t2 = time.time()
    print(f"Total time spent: {(t2 - t1)/60:.4f} minutes")
