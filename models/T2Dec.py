import torch
import torch.nn as nn
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal
import torch.nn.functional as F

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [B, L, N]
        batch_size, seq_len, num_nodes = x.size()
        kernel_size = self.moving_avg.kernel_size[0]
        
        front = x[:, 0:1, :].repeat(1, (kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, kernel_size // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        
        # [B, L, N] -> [B, N, L]
        x_trend = self.moving_avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        x_detail = x - x_trend
        return x_trend, x_detail

# ======================
# 实验三：正交解耦对齐适配器 (Orthogonal Decoupling Alignment)
# ======================
class OrthogonalDecouplingAdapter(nn.Module):
    """
    针对 统计提示词文本 + 时序特征 的正交解耦适配器
    目的：剥离时序特征中与统计文本重叠的宏观信息，保留纯粹的微观动态规律进行互补融合。
    """
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        
        self.ts_norm = nn.LayerNorm(channel)
        self.cross_norm = nn.LayerNorm(channel)

    def forward(self, ts_feat, cross_feat):
        # ts_feat: [B, C, N] (原始时序微观特征)
        # cross_feat: [B, C, N] (统计文本宏观特征)
        ts_h = ts_feat.permute(0, 2, 1)
        cross_h = cross_feat.permute(0, 2, 1)

        ts_h_norm = self.ts_norm(ts_h)
        cross_h_norm = self.cross_norm(cross_h)

        dot_product = torch.sum(ts_h_norm * cross_h_norm, dim=-1, keepdim=True)
        norm_cross_sq = torch.sum(cross_h_norm * cross_h_norm, dim=-1, keepdim=True) + 1e-8
        proj_coeff = dot_product / norm_cross_sq
        ts_proj = proj_coeff * cross_h

        ts_orthogonal = ts_h - ts_proj
        ortho_cos = F.cosine_similarity(ts_orthogonal, cross_h, dim=-1).abs().mean()

        return ts_orthogonal.permute(0, 2, 1), ortho_cos


class T2Dec(nn.Module):
    def __init__(
        self,
        device = "cuda:7",
        channel = 32,
        num_nodes = 7,
        seq_len = 96,
        pred_len = 96,
        dropout_n = 0.1,
        d_llm = 768,
        e_layer = 1,
        d_layer = 1,
        d_ff=32,
        head =8
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n= dropout_n
        self.d_llm = d_llm
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.d_ff = d_ff
        self.head = head

        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)

        self.ts_encoder_layer = nn.TransformerEncoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, 
                                                           norm_first = True,dropout = self.dropout_n).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers = self.e_layer).to(self.device)

        self.prompt_encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_llm, nhead = self.head, batch_first=True, 
                                                               norm_first = True,dropout = self.dropout_n).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers = self.e_layer).to(self.device)

        # ======================
        # 🔥 增强版交互逻辑 (Scenario 1)
        # 1. 文本特征投影层
        # 2. 时频增强型 CrossModal
        # ======================
        self.text_proj = nn.Linear(self.d_llm, self.channel).to(self.device)
        self.cross = CrossModal(d_model=self.channel, n_heads=head, d_ff=self.d_ff, dropout=self.dropout_n).to(self.device)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, norm_first = True, dropout = self.dropout_n).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = self.d_layer).to(self.device)

        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

        # ======================
        # 加入适配器（正交解耦型）
        # ======================
        self.adapter = OrthogonalDecouplingAdapter(channel=self.channel).to(self.device)

        # ======================
        # 新增：动态融合门控 (Dynamic Fusion Gate)
        # 根据文本特征生成权重，动态调节波动分量的贡献度
        # ======================
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.channel * 2, self.channel // 2),  # 输入改成 2*C
            nn.ReLU(),
            nn.Linear(self.channel // 2, 1),
            nn.Sigmoid()
        ).to(self.device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_data, input_data_mark, embeddings):
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()
        embeddings = embeddings.float()

        input_data = self.normalize_layers(input_data, 'norm')

        input_data = input_data.permute(0,2,1)
        input_data = self.length_to_feature(input_data)

        embeddings = embeddings.squeeze(-1)
        # embeddings = embeddings.permute(0,2,1)
        if embeddings.dim() == 2:
            # Handle single variable case: [B, D_LLM] -> [B, 1, D_LLM]
            embeddings = embeddings.unsqueeze(1)
        else:
            # Handle multi variable case: [B, D_LLM, N] -> [B, N, D_LLM]
            embeddings = embeddings.permute(0, 2, 1)
        enc_out = self.ts_encoder(input_data)
        enc_out = enc_out.permute(0,2,1)
        embeddings = self.prompt_encoder(embeddings)
        embeddings = embeddings.permute(0,2,1)

        # ======================
        # 🔥 增强版交互逻辑 (Scenario 1)
        # ======================
        text_feat = self.text_proj(embeddings.permute(0, 2, 1)) 
        
        ts_q = enc_out.permute(0, 2, 1)  # [B, N, C]
        cross_out = self.cross(ts_q, text_feat, text_feat)  # [B, N, C]
        cross_out = cross_out.permute(0, 2, 1)  # [B, C, N]
        
        # ======================
        # 前向干预：正交解耦
        # ======================
        micro_feat, ortho_loss = self.adapter(enc_out, cross_out) 
        
        # ===== Branch =====
        enc_out_perm = enc_out.permute(0, 2, 1)
        micro_feat_perm = micro_feat.permute(0, 2, 1)
        cross_perm = cross_out.permute(0, 2, 1)

        # Trend（语义）
        trend_feat = self.decoder(text_feat, cross_perm)
        y_trend = self.c_to_length(trend_feat)

        # Detail（残差）
        y_detail = self.c_to_length(micro_feat_perm)

        # Fusion（动态）：让 alpha 控制文本趋势的信任度
        # ======================
        fusion_input = torch.cat([text_feat, enc_out_perm], dim=-1)
        alpha = self.fusion_gate(fusion_input) 

        # 分层预测结果：采用加权平均 (alpha 动态平衡)
        dec_out = alpha * y_trend + (1 - alpha) * y_detail
        
        # 保存 loss
        self.ortho_loss = ortho_loss
        
        # 保存 alpha 以供可视化分析（可选）
        self.dynamic_alpha = alpha.detach()
        
        # 记录分层分量用于监督训练：必须与 dec_out 中的项完全对应
        self.pred_trend_norm = alpha * y_trend
        self.pred_detail_norm = (1 - alpha) * y_detail
        
        dec_out = dec_out.permute(0, 2, 1) # [B, N, Pred_Len]
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out, self.pred_trend_norm.permute(0, 2, 1), self.pred_detail_norm.permute(0, 2, 1)
