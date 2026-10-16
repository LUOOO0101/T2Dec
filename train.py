import torch
import re
from torch import optim
import numpy as np
import argparse
import time
import os
import random
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from models.T2Dec import T2Dec, series_decomp
from utils.metrics import MSE, MAE, metric
import faulthandler

faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:150"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1", help="data file name")
    parser.add_argument("--root_path", type=str, default="./datasets/", help="root path of the data file")
    parser.add_argument("--channel", type=int, default=32, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=96, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=96, help="out_len")  # defaut=96
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--dropout_n", type=float, default=0.2, help="dropout rate of neural network layers")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--d_layer", type=int, default=1, help="layers of transformer decoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay rate")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="gpt2", help="llm")
    parser.add_argument("--epochs", type=int, default=30, help="maximum training epochs")
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument(
        "--es_patience",
        type=int,
        default=5,
        help="early stop after this many validation epochs without improvement",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
        help="save path",
    )
    parser.add_argument("--kernel_size", type=int, default=25, help="kernel size for series decomposition")
    parser.add_argument("--gauss_center", type=float, default=264.0, help="center for Gaussian loss weighting")
    parser.add_argument("--gauss_sigma", type=float, default=100.0, help="sigma for Gaussian loss weighting")
    parser.add_argument("--target", type=str, default="OT", help="target feature in S or MS task")
    return parser.parse_args()


class trainer:
    def __init__(
            self,
            scaler,
            channel,
            num_nodes,
            seq_len,
            pred_len,
            dropout_n,
            d_llm,
            e_layer,
            d_layer,
            head,
            lrate,
            wdecay,
            device,
            epochs,
            kernel_size,
            gauss_center,
            gauss_sigma
    ):
        self.model = T2Dec(
            device=device, channel=channel, num_nodes=num_nodes, seq_len=seq_len, pred_len=pred_len,
            dropout_n=dropout_n, d_llm=d_llm, e_layer=e_layer, d_layer=d_layer, head=head
        ).to(device)
        
        self.decomp = series_decomp(kernel_size=kernel_size).to(device) # 真值分解器
        self.gauss_center = gauss_center
        self.gauss_sigma = gauss_sigma

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=min(epochs, 50), eta_min=1e-6)
        self.loss = MSE
        self.MAE = MAE
        self.clip = 5
        print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))
        print("The number of parameters: {}".format(self.model.param_num()))
        # print(self.model)
# 冻结主干，只训练 adapter
        for param in self.model.parameters():
            param.requires_grad = True

        for param in self.model.adapter.parameters():
            param.requires_grad = True

    def train(self, input, mark, embeddings, real):
        self.model.train()
        self.optimizer.zero_grad()
        
        # 1. 模型预测 (同时获取分量)
        predict, pred_trend_norm, pred_detail_norm = self.model(input, mark, embeddings)
        
        # 2. 真值分解 (需要先将 real 归一化，使其与 pred_trend/detail 尺度一致)
        # real: [B, Pred_Len, N]
        real_norm = self.model.normalize_layers(real, 'norm')
        gt_trend_norm, gt_detail_norm = self.decomp(real_norm)
        
        # 3. 计算分层损失
        loss_total = self.loss(predict, real) # 最终输出的 MSE
        loss_trend = self.loss(pred_trend_norm, gt_trend_norm) # 趋势拟合
        loss_detail = self.loss(pred_detail_norm, gt_detail_norm) # 细节拟合
        
        # --- 比例自动缩放 (Scaling by Pred_Len) ---
        # 使用高斯核函数：在 gauss_center 附近权重最大，向两端衰减
        # 这是一种“尺度感知”的 Loss 设计，比硬编码 if-else 更具泛化性
        lambda_hier = 0.1 + 0.4 * np.exp(-((self.model.pred_len - self.gauss_center)**2) / (2 * self.gauss_sigma**2))
        # lambda_hier=0
        loss = loss_total + lambda_hier * (loss_trend + loss_detail)
        
        rec_loss = getattr(self.model, "rec_loss", None)
        if rec_loss is not None:
            loss = loss + 0.1 * rec_loss
        ortho_loss = getattr(self.model, "ortho_loss", None)
        if ortho_loss is not None:
            loss = loss + 0.01 * ortho_loss
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = self.MAE(predict, real)
        return loss.item(), mae.item()

    def eval(self, input, mark, embeddings, real_val):
        self.model.eval()
        with torch.no_grad():
            predict, _, _ = self.model(input, mark, embeddings)  # Extract dec_out
        loss = self.loss(predict, real_val)
        mae = self.MAE(predict, real_val)
        return loss.item(), mae.item()


def load_data(args):
    data_map = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    data_class = data_map.get(args.data_path, Dataset_Custom)
    train_set = data_class(root_path=args.root_path, flag='train', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, target=args.target)
    val_set = data_class(root_path=args.root_path, flag='val', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, target=args.target)
    test_set = data_class(root_path=args.root_path, flag='test', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, target=args.target)

    scaler = train_set.scaler

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
                             num_workers=args.num_workers)

    # return train_loader, val_loader, test_loader, scaler
    return train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def main():
    args = parse_args()
    train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler = load_data(args)

    print()
    seed_it(args.seed)
    # device = torch.device(args.device)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    loss = 9999999
    epochs_since_best_mse = 0
    bestid = 0
    # 生成路径
    # 修复：只对文件名/文件夹名中的非法字符进行替换，保留路径分隔符
    folder_name = (
        f"{args.pred_len}_{args.channel}_{args.e_layer}_{args.d_layer}_"
        f"{args.learning_rate}_{args.dropout_n}_bs{args.batch_size}_{args.target}_{args.seed}"
    )
    # 替换非法字符（如 Windows 下的 : 等），但保留 / 路径分隔符
    safe_save = args.save.replace(':', '_')
    path = os.path.join(safe_save, args.data_path, folder_name)
    path = path.rstrip('/') + '/' # 确保以 / 结尾以便后续拼接文件名

    # 确保路径存在
    if not os.path.exists(path):
        os.makedirs(path)

    his_loss = []
    val_time = []
    train_time = []
    print(args)

    engine = trainer(
        scaler=scaler,
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        dropout_n=args.dropout_n,
        d_llm=args.d_llm,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        head=args.head,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        device=device,
        epochs=args.epochs,
        kernel_size=args.kernel_size,
        gauss_center=args.gauss_center,
        gauss_sigma=args.gauss_sigma
    )

    print("Start training...", flush=True)

    for i in range(1, args.epochs + 1):
        t1 = time.time()
        train_loss = []
        train_mae = []

        for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(train_loader):
            trainx = torch.as_tensor(x, dtype=torch.float32, device=device)  # [B, L, N]
            trainy = torch.as_tensor(y, dtype=torch.float32, device=device)
            trainx_mark = torch.as_tensor(x_mark, dtype=torch.float32, device=device)
            train_embedding = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
            metrics = engine.train(trainx, trainx_mark, train_embedding, trainy)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])

        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        # validation
        val_loss = []
        val_mae = []
        s1 = time.time()

        for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(val_loader):
            valx = torch.as_tensor(x, dtype=torch.float32, device=device)
            valy = torch.as_tensor(y, dtype=torch.float32, device=device)
            valx_mark = torch.as_tensor(x_mark, dtype=torch.float32, device=device)
            val_embedding = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
            metrics = engine.eval(valx, valx_mark, val_embedding, valy)
            val_loss.append(metrics[0])
            val_mae.append(metrics[1])

        s2 = time.time()
        log = "Epoch: {:03d}, Validation Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mvalid_loss = np.mean(val_loss)
        mvalid_mae = np.mean(val_mae)

        his_loss.append(mvalid_loss)
        print("-----------------------")

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f} "
        print(
            log.format(i, mtrain_loss, mtrain_mae),
            flush=True,
        )
        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid MAE: {:.4f}"
        print(
            log.format(i, mvalid_loss, mvalid_mae),
            flush=True,
        )

        if mvalid_loss < loss:
            print("###Update tasks appear###")
            loss = mvalid_loss
            torch.save(engine.model.state_dict(), path + "best_model.pth")
            bestid = i
            epochs_since_best_mse = 0
            print("Updating! Valid Loss:{:.4f}".format(mvalid_loss), end=", ")
            print("epoch: ", i)
        else:
            epochs_since_best_mse += 1
            print("No update")

        engine.scheduler.step()

        if epochs_since_best_mse >= args.es_patience:
            print("Early stopping after {} epochs without validation improvement.".format(args.es_patience))
            break

    # Output consumption
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Validation Time: {:.4f} secs".format(np.mean(val_time)))

    # Test
    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))

    engine.model.load_state_dict(torch.load(path + "best_model.pth"))

    test_outputs = []
    test_y = []

    for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(test_loader):
        testx = torch.as_tensor(x, dtype=torch.float32, device=device)
        testy = torch.as_tensor(y, dtype=torch.float32, device=device)
        testx_mark = torch.as_tensor(x_mark, dtype=torch.float32, device=device)
        test_embedding = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
        with torch.no_grad():
            preds, _, _ = engine.model(testx, testx_mark, test_embedding)
        test_outputs.append(preds)
        test_y.append(testy)

    test_pre = torch.cat(test_outputs, dim=0)
    test_real = torch.cat(test_y, dim=0)

    amse = []
    amae = []
    
    # 准备保存结果的列表
    res_logs = []
    res_logs.append(f"The epoch of the best result: {bestid}\n")
    res_logs.append(f"The valid loss of the best model: {round(his_loss[bestid - 1], 4)}\n\n")

    for j in range(args.pred_len):
        pred = test_pre[:, j, ].to(device)
        real = test_real[:, j, ].to(device)
        metrics = metric(pred, real)
        log = "Evaluate best model on test data for horizon {:d}, Test MSE: {:.4f}, Test MAE: {:.4f}"
        print(log.format(j, metrics[0], metrics[1]))
        res_logs.append(log.format(j, metrics[0], metrics[1]) + "\n")
        amse.append(metrics[0])
        amae.append(metrics[1])

    avg_log = "On average horizons, Test MSE: {:.4f}, Test MAE: {:.4f}"
    print(avg_log.format(np.mean(amse), np.mean(amae)))
    res_logs.append("\n" + avg_log.format(np.mean(amse), np.mean(amae)) + "\n")
    
    # 将结果写入文件
    with open(os.path.join(path, "test_results.txt"), "w") as f:
        f.writelines(res_logs)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
