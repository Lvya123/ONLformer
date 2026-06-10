# # from utils.data_RGB import get_validation_data
# # from skimage import img_as_ubyte
# # from skimage.metrics import peak_signal_noise_ratio as PSNR
# # import numpy as np
# # import os
# # import torch.nn as nn
# # import torch
# # from torch.utils.data import DataLoader
# # import torch.nn.functional as F
# # import utils
# # from logger import *
# # import yaml
# # with open('test.yml', mode='r') as f_yml:
# #     Loader, _ = ordered_yaml()
# #     opt = yaml.load(f_yml, Loader=Loader)

# # gpus = ','.join([str(i) for i in opt['GPU']])

# # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # os.environ["CUDA_VISIBLE_DEVICES"] = gpus

# # model_restoration = utils.get_arch(opt['MODEL'])
# # model_restoration.eval().cuda()

# # dir_name = os.path.dirname(os.path.abspath(__file__))
# # log_dir = os.path.join(dir_name, 'log', opt['MODEL']['NAME'] + '_' + opt['MODEL']['MODE'])
# # model_dir  = os.path.join(log_dir, 'models')
# # path_chk_rest = os.path.join(model_dir, opt['VAL']['PRETRAIN_MODEL'])
# # featuremap_dir = os.path.join(dir_name, 'featuremaps')
# # utils.load_checkpoint(model_restoration, path_chk_rest)
# # val_epoch = utils.load_start_epoch(path_chk_rest)

# # val_dataset = get_validation_data(opt['PATH']['VAL_DATASET'], {'patch_size':opt['VAL']['VAL_PS']})
# # val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

# # out_list = []
# # inp_list = []
# # def forward_hooki(model, input, output):
# #     inp_list.append(input)

# # def forward_hooko(model, input, output):
# #     out_list.append(output)

# # layer_dir = os.path.join(featuremap_dir, 'attention_map')
# # hooki = model_restoration.Layers[5].layers[3].attention
# # hooki.register_forward_hook(forward_hooki)

# # hooko = model_restoration.Layers[5].layers[3].attention
# # hooko.register_forward_hook(forward_hooko)

# # print(hooki)
# # print('--------------------------')
# # print(hooko)

# # # exit(0)
# # for ii, data_val in enumerate(val_loader, 1):
# #     img_path = os.path.join(layer_dir, str(ii))
# #     utils.mkdirs(img_path)

# #     if ii == 4:
# #         input_ = data_val[1].cuda()
# #         with torch.no_grad():
# #             out_list = []
# #             inp_list = []

# #             input_ = data_val[1].cuda()
# #             # factor = 32
# #             # h,w = input_.shape[2],input_.shape[3]
# #             # H,W = ((h+factor)//factor)*factor,((w+factor)//factor)*factor
# #             # padh = H-h if h%factor!=0 else 0
# #             # padw = W-w if w%factor!=0 else 0
# #             # input_ = F.pad(input_,(0,padw,0,padh),'reflect')

# #             # print(input_.shape)

# #             restored = model_restoration(input_)[-1]

# #             # print('restored', restored.shape)

# #             # break
# #             # restored = restored[...,:h,:w]
# #             featuremap = out_list[0]
# #             featuremap_in = inp_list[0]
            
# #             # print(featuremap.shape)
# #             # break
# #             # featuremapi = inp_list[0][0]
# #             # featuremapo = out_list[0]
# #             # featuremap = featuremapi + featuremapo
# #             featuremap = img_as_ubyte(torch.clamp(featuremap + featuremap_in[0],0,1).cpu().squeeze().numpy().transpose(1,2,0))

# #             del input_
# #             del restored
# #             del out_list 
# #             del inp_list 

# #             torch.cuda.empty_cache()

# #         h,w,c = featuremap.shape
# #         for i in range(c):
# #             file_name = str(i)+'.png'
# #             # utils.mkdirs(layer_dir)
# #             utils.save_img(os.path.join(img_path, file_name), featuremap[:,:,i])
# #         print(ii, 'finished')
# #     # break
















# # from utils.data_RGB import get_validation_data
# # import numpy as np
# # import os
# # import torch
# # from torch.utils.data import DataLoader
# # import utils
# # from logger import *
# # import yaml

# # def compute_spatial_variance_fast(feat, topk_ratio=0.25):
# #     """
# #     快速计算空间方差（批量处理）
# #     坐标归一化到[0,1]范围
# #     feat: [C, H, W]
# #     """
# #     C, H, W = feat.shape
    
# #     fmap_flat = feat.view(C, -1)
# #     K = max(1, int(topk_ratio * H * W))
    
# #     topk_vals, topk_idx = torch.topk(fmap_flat, K, dim=1)
    
# #     # 计算坐标并归一化到[0,1]
# #     ys = (topk_idx // W).float() / (H - 1)  # 归一化y坐标
# #     xs = (topk_idx % W).float() / (W - 1)   # 归一化x坐标
    
# #     x_mean = xs.mean(dim=1, keepdim=True)
# #     y_mean = ys.mean(dim=1, keepdim=True)
    
# #     var = ((xs - x_mean) ** 2 + (ys - y_mean) ** 2).mean(dim=1)
    
# #     return var.mean()

# # def compute_entropy_fast(feat, eps=1e-8):
# #     """
# #     快速计算熵（批量处理）
# #     feat: [C, H, W]
# #     """
# #     C, H, W = feat.shape
    
# #     fmap_flat = feat.view(C, -1)
    
# #     fmap_min = fmap_flat.min(dim=1, keepdim=True)[0]
# #     p = fmap_flat - fmap_min
# #     p_sum = p.sum(dim=1, keepdim=True)
# #     p = p / (p_sum + eps)
    
# #     entropy = -(p * torch.log(p + eps)).sum(dim=1)
    
# #     return entropy.mean()

# # # 读取配置
# # with open('test.yml', mode='r') as f_yml:
# #     Loader, _ = ordered_yaml()
# #     opt = yaml.load(f_yml, Loader=Loader)

# # gpus = ','.join([str(i) for i in opt['GPU']])
# # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # os.environ["CUDA_VISIBLE_DEVICES"] = gpus

# # model_restoration = utils.get_arch(opt['MODEL'])
# # model_restoration.eval().cuda()

# # dir_name = os.path.dirname(os.path.abspath(__file__))
# # log_dir = os.path.join(dir_name, 'log', opt['MODEL']['NAME'] + '_' + opt['MODEL']['MODE'])
# # path_chk_rest = r"/media/ubutu/LYN吕亚楠/onlformer/log/UNet_small_baseline/models/model_epoch_2400.pth"
# # utils.load_checkpoint(model_restoration, path_chk_rest)
# # val_epoch = utils.load_start_epoch(path_chk_rest)

# # val_dataset = get_validation_data(opt['PATH']['VAL_DATASET'], {'patch_size':opt['VAL']['VAL_PS']})
# # val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

# # out_list = []
# # inp_list = []

# # def forward_hooki(model, input, output):
# #     inp_list.append(input)

# # def forward_hooko(model, input, output):
# #     out_list.append(output)

# # hooki = model_restoration.Layers[5].layers[3].attention
# # hooki.register_forward_hook(forward_hooki)
# # hooko = model_restoration.Layers[5].layers[3].attention
# # hooko.register_forward_hook(forward_hooko)

# # all_spatial_variances = []
# # all_entropies = []

# # print("Processing images...")
# # print("-" * 60)

# # for ii, data_val in enumerate(val_loader, 1):
# #     out_list = []
# #     inp_list = []
    
# #     input_ = data_val[1].cuda()
# #     with torch.no_grad():
# #         restored = model_restoration(input_)[-1]
        
# #         if len(out_list) > 0 and len(inp_list) > 0:
# #             featuremap = out_list[0]
# #             featuremap_in = inp_list[0]
            
# #             # 残差连接
# #             # attention_map = featuremap + featuremap_in[0]
# #             attention_map = featuremap

            
# #             # 获取attention map的尺寸
# #             C, H, W = attention_map[0].shape
# #             # print(f"Attention map size: {H}×{W}, channels: {C}")
            
# #             # 计算指标
# #             spatial_var = compute_spatial_variance_fast(attention_map[0], topk_ratio=0.25)
# #             entropy = compute_entropy_fast(attention_map[0])
            
# #             all_spatial_variances.append(spatial_var.item())
# #             all_entropies.append(entropy.item())
            
# #             # 每张图打印
# #             print(f"Image {ii:3d} - Spatial Variance: {spatial_var.item():.6f}, Entropy: {entropy.item():.6f}")
        
# #         del input_, restored, out_list, inp_list
# #         torch.cuda.empty_cache()

# # print("-" * 60)

# # # 打印平均结果
# # if len(all_spatial_variances) > 0:
# #     avg_spatial_var = np.mean(all_spatial_variances)
# #     avg_entropy = np.mean(all_entropies)
    
# #     print(f"\nAverage Results over {len(all_spatial_variances)} images:")
# #     print(f"Average Spatial Variance: {avg_spatial_var:.6f}")
# #     print(f"Average Entropy: {avg_entropy:.6f}")
# # else:
# #     print("No data processed!")





































import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import yaml
import utils
from utils.data_RGB import get_validation_data
from logger import *
from einops import rearrange

# =========================
# ✅ 指标1：Attention Distance
# =========================
def compute_attention_distance(attn, Hq, Wq):

    B, heads, Nq, Nk = attn.shape

    q_idx = torch.arange(Nq, device=attn.device)
    qy = (q_idx // Wq).float()
    qx = (q_idx % Wq).float()

    k_idx = torch.arange(Nk, device=attn.device)
    ky = (k_idx // Wq).float()
    kx = (k_idx % Wq).float()

    qy = qy.view(1, 1, Nq, 1)
    qx = qx.view(1, 1, Nq, 1)
    ky = ky.view(1, 1, 1, Nk)
    kx = kx.view(1, 1, 1, Nk)

    dist = torch.sqrt((qy - ky) ** 2 + (qx - kx) ** 2)

    max_dist = torch.sqrt(torch.tensor(Hq**2 + Wq**2, device=attn.device).float())
    dist = dist / max_dist

    expected_dist = (attn * dist).sum(dim=-1)

    return expected_dist.mean()


# =========================
# ✅ 指标2：Attention Variance
# =========================
# def compute_attention_variance(attn):
#     return attn.var(dim=-1).mean()
def compute_topk_distance_variance(attn, Hq, Wq, topk_ratio=0.25):
    B, heads, Nq, Nk = attn.shape
    k_val = max(1, int(Nk * topk_ratio))

    k_idx = torch.arange(Nk, device=attn.device)
    ky = (k_idx // Wq).float() / Hq
    kx = (k_idx % Wq).float() / Wq

    ky = ky.view(1, 1, 1, Nk).expand(B, heads, Nq, Nk)
    kx = kx.view(1, 1, 1, Nk).expand(B, heads, Nq, Nk)

    topk_idx = torch.topk(attn, k=k_val, dim=-1).indices

    topk_kx = torch.gather(kx, -1, topk_idx)
    topk_ky = torch.gather(ky, -1, topk_idx)

    var_x = topk_kx.var(dim=-1)
    var_y = topk_ky.var(dim=-1)

    return (var_x + var_y).mean()



# =========================
# ✅ 指标3：Attention Entropy（强烈推荐）
# =========================
def compute_attention_entropy(attn):
    eps = 1e-8
    entropy = -(attn * torch.log(attn + eps)).sum(dim=-1)
    return entropy.mean()


# =========================
# ✅ ONL attention 上采样（+归一化！）
# =========================
def upsample_attn(attn, Hq, Wq, pool_size=4):

    B, heads, Nq, Nk = attn.shape

    Hk = Hq // pool_size
    Wk = Wq // pool_size

    attn = attn.view(B, heads, Nq, Hk, Wk)

    attn = attn.repeat_interleave(pool_size, dim=-2)
    attn = attn.repeat_interleave(pool_size, dim=-1)

    attn = attn.view(B, heads, Nq, Hq * Wq)

    # ✅ 关键：重新归一化（必须！）
    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

    return attn


# =========================
# 配置
# =========================
with open('test.yml', 'r') as f:
    Loader, _ = ordered_yaml()
    opt = yaml.load(f, Loader=Loader)

gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_VISIBLE_DEVICES"] = gpus


# =========================
# 模型
# =========================
model = utils.get_arch(opt['MODEL']).cuda()
model.eval()

ckpt = "/media/ubutu/93efb5a9-c4b0-47d4-aadc-68c9dee1eb18/LYN/0/onlformer/log/UNet_small_baseline/models/model_epoch_2400.pth"
utils.load_checkpoint(model, ckpt)


# =========================
# 数据
# =========================
val_dataset = get_validation_data(
    opt['PATH']['VAL_DATASET'],
    {'patch_size': opt['VAL']['VAL_PS']}
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# =========================
# window 分块
# =========================

def window_partitionx(x, window_size):
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partitions(x[:, :, :h, :w], window_size)
    b_main = x_main.shape[0]
    if h == H and w == W:
        return x_main, [b_main]
    if h != H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_r
        x_dd = x[:, :, -window_size:, -window_size:]
        b_dd = x_dd.shape[0] + b_d
        # batch_list = [b_main, b_r, b_d, b_dd]
        return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
    if h == H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
    if h != H and w == W:
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_main
        return torch.cat([x_main, x_d], dim=0), [b_main, b_d]



def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

# =========================
# Hook：重建 attention
# =========================
attn_list = []

def hook_fn(module, input, output):

    x = input[0]

    with torch.no_grad():
        b, c, h, w = x.shape

        x_win, _ = window_partitionx(x, module.win_size)

        qkv = module.qkv(x_win)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # q增强
        q_flatten = torch.flatten(q, -2)
        q_flatten = module.fc(q_flatten)
        q_ = q_flatten.reshape(-1, c, module.win_size, module.win_size)
        q_ = module.sigmoid(q_)
        q = q * q_
        q = module.ln_q(q)

        # ONL关键：pool k
        k = module.maxpool(k)

        # reshape
        q = rearrange(q, 'b (n c) h w -> b n (h w) c', n=module.num_heads)
        k = rearrange(k, 'b (n c) h w -> b n (h w) c', n=module.num_heads)

        # attention
        attn = (q @ k.transpose(-2, -1)) * module.temperature
        attn = attn.softmax(dim=-1)

        attn_list.append(attn.detach())


# 挂 hook
hook = model.Layers[5].layers[3].attention
handle = hook.register_forward_hook(hook_fn)


# =========================
# 主循环
# =========================
all_dist, all_var, all_ent = [], [], []

print("Processing...")
print("-" * 60)

for i, data in enumerate(val_loader, 1):

    if i > 5:   # ✅ 控制样本数量
        break

    attn_list.clear()

    inp = data[1].cuda()

    with torch.no_grad():
        _ = model(inp)

    if len(attn_list) > 0:

        attn = attn_list[0]

        _, _, Nq, Nk = attn.shape

        win = hook.win_size
        Hq = Wq = win

        # ONL 需要上采样
        if Nk < Nq:
            attn = upsample_attn(attn, Hq, Wq, pool_size=4)

        # 三指标
        dist = compute_attention_distance(attn, Hq, Wq)
        # var = compute_attention_variance(attn)
        var = compute_topk_distance_variance(attn, Hq, Wq, topk_ratio=0.4)
        ent = compute_attention_entropy(attn)

        all_dist.append(dist.item())
        all_var.append(var.item())
        all_ent.append(ent.item())

        print(f"Image {i:3d} | Dist: {dist:.4f} | Var: {var:.6f} | Ent: {ent:.4f}")

    torch.cuda.empty_cache()

print("-" * 60)


# =========================
# 平均结果
# =========================
if len(all_dist) > 0:
    print(f"\nAverage over {len(all_dist)} images:")
    print(f"Attention Distance : {np.mean(all_dist):.6f}")
    print(f"Attention Variance : {np.mean(all_var):.6f}")
    print(f"Attention Entropy  : {np.mean(all_ent):.6f}")
else:
    print("No data!")


# =========================
# 清理
# =========================
handle.remove()














# import torch
# import numpy as np
# import os
# import yaml
# import utils

# from torch.utils.data import DataLoader
# from utils.data_RGB import get_validation_data
# from einops import rearrange

# import matplotlib.pyplot as plt





# def window_partitionx(x, window_size):
#     _, _, H, W = x.shape
#     h, w = window_size * (H // window_size), window_size * (W // window_size)
#     x_main = window_partitions(x[:, :, :h, :w], window_size)
#     b_main = x_main.shape[0]
#     if h == H and w == W:
#         return x_main, [b_main]
#     if h != H and w != W:
#         x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
#         b_r = x_r.shape[0] + b_main
#         x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
#         b_d = x_d.shape[0] + b_r
#         x_dd = x[:, :, -window_size:, -window_size:]
#         b_dd = x_dd.shape[0] + b_d
#         # batch_list = [b_main, b_r, b_d, b_dd]
#         return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
#     if h == H and w != W:
#         x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
#         b_r = x_r.shape[0] + b_main
#         return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
#     if h != H and w == W:
#         x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
#         b_d = x_d.shape[0] + b_main
#         return torch.cat([x_main, x_d], dim=0), [b_main, b_d]



# def window_partitions(x, window_size):
#     """
#     Args:
#         x: (B, C, H, W)
#         window_size (int): window size

#     Returns:
#         windows: (num_windows*B, C, window_size, window_size)
#     """
#     B, C, H, W = x.shape
#     x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
#     windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
#     return windows

# # =========================
# # ✅ Attention Distance
# # =========================
# def compute_attention_distance(attn, Hq, Wq):
#     B, heads, Nq, Nk = attn.shape

#     q_idx = torch.arange(Nq, device=attn.device)
#     qy = (q_idx // Wq).float()
#     qx = (q_idx % Wq).float()

#     k_idx = torch.arange(Nk, device=attn.device)
#     ky = (k_idx // Wq).float()
#     kx = (k_idx % Wq).float()

#     qy = qy.view(1, 1, Nq, 1)
#     qx = qx.view(1, 1, Nq, 1)
#     ky = ky.view(1, 1, 1, Nk)
#     kx = kx.view(1, 1, 1, Nk)

#     dist = torch.sqrt((qy - ky) ** 2 + (qx - kx) ** 2)

#     max_dist = torch.sqrt(torch.tensor(Hq**2 + Wq**2, device=attn.device).float())
#     dist = dist / max_dist

#     return (attn * dist).sum(dim=-1).mean()


# # =========================
# # ✅ Top-K variance（你原本）
# # =========================
# def compute_topk_distance_variance(attn, Hq, Wq, topk_ratio=0.25):
#     B, heads, Nq, Nk = attn.shape
#     k_val = max(1, int(Nk * topk_ratio))

#     k_idx = torch.arange(Nk, device=attn.device)
#     ky = (k_idx // Wq).float() / Hq
#     kx = (k_idx % Wq).float() / Wq

#     ky = ky.view(1, 1, 1, Nk).expand(B, heads, Nq, Nk)
#     kx = kx.view(1, 1, 1, Nk).expand(B, heads, Nq, Nk)

#     topk_idx = torch.topk(attn, k=k_val, dim=-1).indices

#     topk_kx = torch.gather(kx, -1, topk_idx)
#     topk_ky = torch.gather(ky, -1, topk_idx)

#     var_x = topk_kx.var(dim=-1)
#     var_y = topk_ky.var(dim=-1)

#     return (var_x + var_y).mean()


# # =========================
# # ✅ Entropy
# # =========================
# def compute_attention_entropy(attn):
#     eps = 1e-8
#     return (-(attn * torch.log(attn + eps)).sum(dim=-1)).mean()


# # =========================
# # ⭐ NEW1: Far-range ratio
# # =========================
# def compute_far_range_ratio(attn, Hq, Wq, tau=0.5):
#     B, heads, Nq, Nk = attn.shape

#     q_idx = torch.arange(Nq, device=attn.device)
#     qy = (q_idx // Wq).float()
#     qx = (q_idx % Wq).float()

#     k_idx = torch.arange(Nk, device=attn.device)
#     ky = (k_idx // Wq).float()
#     kx = (k_idx % Wq).float()

#     qy = qy.view(1, 1, Nq, 1)
#     qx = qx.view(1, 1, Nq, 1)
#     ky = ky.view(1, 1, 1, Nk)
#     kx = kx.view(1, 1, 1, Nk)

#     dist = torch.sqrt((qy - ky) ** 2 + (qx - kx) ** 2)

#     max_dist = torch.sqrt(torch.tensor(Hq**2 + Wq**2, device=attn.device).float())
#     dist = dist / max_dist

#     mask_far = (dist > tau).float()

#     return (attn * mask_far).sum(dim=-1).mean()


# # =========================
# # ⭐ NEW2: Top-K weighted distance
# # =========================
# def compute_topk_weighted_distance(attn, Hq, Wq, topk_ratio=0.2):
#     B, heads, Nq, Nk = attn.shape
#     k_val = max(1, int(Nk * topk_ratio))

#     q_idx = torch.arange(Nq, device=attn.device)
#     qy = (q_idx // Wq).float()
#     qx = (q_idx % Wq).float()

#     k_idx = torch.arange(Nk, device=attn.device)
#     ky = (k_idx // Wq).float()
#     kx = (k_idx % Wq).float()

#     qy = qy.view(1, 1, Nq, 1)
#     qx = qx.view(1, 1, Nq, 1)
#     ky = ky.view(1, 1, 1, Nk)
#     kx = kx.view(1, 1, 1, Nk)

#     dist = torch.sqrt((qy - ky) ** 2 + (qx - kx) ** 2)

#     max_dist = torch.sqrt(torch.tensor(Hq**2 + Wq**2, device=attn.device).float())
#     dist = dist / max_dist

#     topk_idx = torch.topk(attn, k=k_val, dim=-1).indices
#     topk_attn = torch.gather(attn, -1, topk_idx)
#     topk_dist = torch.gather(dist, -1, topk_idx)

#     return (topk_attn * topk_dist).sum(dim=-1).mean()


# # =========================
# # ⭐ NEW3: Decay curve
# # =========================
# def compute_distance_bins(attn, Hq, Wq, num_bins=10):
#     B, heads, Nq, Nk = attn.shape

#     q_idx = torch.arange(Nq, device=attn.device)
#     k_idx = torch.arange(Nk, device=attn.device)

#     qy = (q_idx // Wq).float()
#     qx = (q_idx % Wq).float()
#     ky = (k_idx // Wq).float()
#     kx = (k_idx % Wq).float()

#     dist = torch.sqrt(
#         (qy.view(1,1,Nq,1) - ky.view(1,1,1,Nk))**2 +
#         (qx.view(1,1,Nq,1) - kx.view(1,1,1,Nk))**2
#     )

#     dist = dist / dist.max()

#     bins = torch.linspace(0, 1, num_bins+1, device=attn.device)

#     curve = []
#     for i in range(num_bins):
#         mask = (dist >= bins[i]) & (dist < bins[i+1])
#         curve.append((attn * mask.float()).mean().item())

#     return np.array(curve)


# # =========================
# # ONL upsample
# # =========================
# def upsample_attn(attn, Hq, Wq, pool_size=4):
#     B, heads, Nq, Nk = attn.shape

#     Hk = Hq // pool_size
#     Wk = Wq // pool_size

#     attn = attn.view(B, heads, Nq, Hk, Wk)

#     attn = attn.repeat_interleave(pool_size, dim=-2)
#     attn = attn.repeat_interleave(pool_size, dim=-1)

#     attn = attn.view(B, heads, Nq, Hq * Wq)

#     return attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)


# # =========================
# # config
# # =========================
# with open('test.yml', 'r') as f:
#     opt = yaml.load(f, Loader=yaml.SafeLoader)

# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in opt['GPU']])


# # =========================
# # model
# # =========================
# model = utils.get_arch(opt['MODEL']).cuda()
# model.eval()

# ckpt = "/media/ubutu/93efb5a9-c4b0-47d4-aadc-68c9dee1eb18/LYN/0/onlformer/log/UNet_small_baseline/models/model_epoch_2400.pth"
# utils.load_checkpoint(model, ckpt)


# # =========================
# # data
# # =========================
# val_dataset = get_validation_data(opt['PATH']['VAL_DATASET'],
#                                  {'patch_size': opt['VAL']['VAL_PS']})

# val_loader = DataLoader(val_dataset,
#                         batch_size=1,
#                         shuffle=False,
#                         num_workers=4,
#                         pin_memory=True)


# # =========================
# # hook
# # =========================
# attn_list = []

# def hook_fn(module, input, output):
#     x = input[0]

#     with torch.no_grad():
#         b, c, h, w = x.shape
#         x_win, _ = window_partitionx(x, 32)
#         qkv = module.qkv(x_win)
#         q, k, v = torch.chunk(qkv, 3, dim=1)
#         # ... (Q/K 的增强和 LN 操作保持不变) ...
        
#         # ONL关键：pool k
#         k = module.maxpool(k)

#         # reshape
#         q = rearrange(q, 'b (n c) h w -> b n (h w) c', n=module.num_heads)
#         k = rearrange(k, 'b (n c) h w -> b n (h w) c', n=module.num_heads)

#         # ⚡ 关键优化：不要显式计算完整的 attn 矩阵 [B, H, Nq, Nk]
#         # =========================================================
#         # 方法1（推荐）：仅计算你需要的 metrics
#         # 这需要你重写指标函数，使其能在不展开完整矩阵的情况下计算
#         # =========================================================
        
#         # 为了调试，我们先打印 shape 并降级计算 Distance（如果空间允许）
#         B, heads, Nq, c_dim = q.shape
#         Nk = k.shape[2]
#         print(f"Shape in hook: q={q.shape}, k={k.shape} -> attn would be [{B},{heads},{Nq},{Nk}]")
        
#         # ⚠️ 这里才尝试计算 attn，但会 OOM
#         # 注释掉下面这行以继续运行
#         # attn = (q @ k.transpose(-2, -1)) * module.temperature
        
#         # 替代：只计算一个示例或采样一部分 heads/batch
#         # 例如，只取第一个 batch 的第一个 head 来计算指标
#         q_sample = q[0:1, 0:1, :, :]          # [1, 1, Nq, C]
#         k_sample = k[0:1, 0:1, :, :]          # [1, 1, Nk, C]
#         attn_sample = (q_sample @ k_sample.transpose(-2, -1)) * module.temperature[0]
#         # attn_sample shape: [1, 1, Nq, Nk] -> 对于 1024x64 仍是 262K 个元素 (约 1MB)，可接受
#         attn_sample = attn_sample.softmax(dim=-1)  # 注意：这里 softmax 是在 Nk=64 上
        
#         # 现在可以用 attn_sample 计算一个近似指标
#         # 但由于 Nk != Nq (64 vs 1024)，你需要后续上采样
        
#         # 将这个采样结果保存到全局列表
#         attn_list.append(attn_sample.detach())


# handle = model.Layers[5].layers[3].attention.register_forward_hook(hook_fn)


# # =========================
# # run
# # =========================
# all_dist, all_var, all_ent = [], [], []
# all_far, all_topk, all_curves = [], [], []

# for i, data in enumerate(val_loader):

#     if i > 5:
#         break

#     attn_list.clear()

#     inp = data[1].cuda()

#     with torch.no_grad():
#         _ = model(inp)

#     attn = attn_list[0]

#     _, _, Nq, Nk = attn.shape
#     win = model.Layers[5].layers[3].attention.win_size
#     Hq = Wq = win

#     if Nk < Nq:
#         attn = upsample_attn(attn, Hq, Wq)

#     dist = compute_attention_distance(attn, Hq, Wq)
#     var = compute_topk_distance_variance(attn, Hq, Wq)
#     ent = compute_attention_entropy(attn)

#     far = compute_far_range_ratio(attn, Hq, Wq)
#     topk_wd = compute_topk_weighted_distance(attn, Hq, Wq)
#     curve = compute_distance_bins(attn, Hq, Wq)

#     all_dist.append(dist.item())
#     all_var.append(var.item())
#     all_ent.append(ent.item())
#     all_far.append(far.item())
#     all_topk.append(topk_wd.item())
#     all_curves.append(curve)

#     print(f"Dist:{dist:.4f} Var:{var:.4f} Ent:{ent:.4f} Far:{far:.4f} TopK-D:{topk_wd:.4f}")


# # =========================
# # summary
# # =========================
# print("\n===== FINAL RESULTS =====")
# print("Distance :", np.mean(all_dist))
# print("Variance :", np.mean(all_var))
# print("Entropy  :", np.mean(all_ent))
# print("Far Ratio:", np.mean(all_far))
# print("TopK-D   :", np.mean(all_topk))


# # =========================
# # decay curve plot
# # =========================
# curves = np.mean(np.stack(all_curves, 0), 0)

# plt.figure()
# plt.plot(curves)
# plt.xlabel("Distance bin")
# plt.ylabel("Attention weight")
# plt.title("Attention Decay Curve")
# plt.grid()
# plt.show()


# handle.remove()