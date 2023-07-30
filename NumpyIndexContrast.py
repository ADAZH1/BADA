import torch
a1 = torch.Tensor([ # 源域特征矩阵
    [1, 2, 3, 4],
    [4,5,6,7],
    [3, 4, 5, 6],
    [2, 3, 4, 5],
])
a2 = torch.Tensor([ # 目标域特征矩阵
    [8,9,10,11],
    [14,15,16,17],
    [11,12,13,14],
])

matrix = torch.cdist(a1, a2)
matrix = matrix + 1.0
matrix = 1.0/matrix

ind = torch.sort(matrix, descending=True, dim=0).indices
ind_split = torch.split(ind, 1, dim=1)

ind_split = [id.squeeze() for id in ind_split]


sim_matrix_split = torch.split(matrix, 1, dim=1)
sim_matrix_split = [_id.squeeze() for _id in sim_matrix_split]
print(sim_matrix_split)

tgt_lab = torch.Tensor([0, 3, 1])
src_labels = torch.Tensor([2,1,3,0])
vr_src = src_labels.unsqueeze(-1).repeat(1, 4)
label_list = []
sim_score = 0.0
for i in range(0,3):
    _row = ind_split[i].long()
    sim_scoreTemp = 0.0
    unsim_scoreTemp = 0.0
    for j in range(0,2):
        if src_labels[_row[j].item()].item() == tgt_lab[i].item():
            sim_scoreTemp += sim_matrix_split[i][_row[j].item()].item()
        else:
            unsim_scoreTemp += sim_matrix_split[i][_row[j].item()].item()
        ratio = sim_scoreTemp / unsim_scoreTemp
        confidence.append(ratio)
print(len(ind_split))
print(sim_score)
src_labels = torch.rand(2427)
print(src_labels)

vr_src = src_labels.unsqueeze(-1).repeat(1, 4439)
label_list = []
for i in range(0, 4439):
    _row = ind_split[i].long()  # 第i个源域样本在目标域最近的编号  不对
    print(_row)
    #  上面dim=0排序之后  应该是第i个目标域样本在源域中距离最近的样本编号
    _col = (torch.ones(2427) * i).long()
    _val = vr_src[_row, _col]  # val出来的是 根据上面那些下标在

    top_n_val = _val[[j for j in range(0, 4)]]
    label_list.append(top_n_val)

    all_top_labels = torch.stack(label_list, dim=1)
assigned_tgt_labels = torch.mode(all_top_labels, dim=0).values
flat_src_labels = src_labels.squeeze()

sim_matrix_split = torch.split(matrix, 1, dim=1)
sim_matrix_split = [_id.squeeze() for _id in sim_matrix_split]

simratio_score = []  # sim-ratio (knn conf measure) for all tgt

for i in range(0, 4439):  # nln: nearest like neighbours, nun: nearest unlike neighbours
    t_label = assigned_tgt_labels[i]
    nln_mask = (flat_src_labels == t_label)

    nun_mask = ~(flat_src_labels == t_label)
    nun_sim_all = sim_matrix_split[i][nun_mask]

    len1 = len(nun_sim_all)
    nun_sim_r = torch.narrow(torch.sort(nun_sim_all, descending=True)[0], 0, 0, len(nun_sim_all))

    nln_sim_all = sim_matrix_split[i][nln_mask]
    nln_sim_r = torch.narrow(torch.sort(nln_sim_all, descending=True)[0], 0, 0, len(nln_sim_all))

    nln_sim_score = 1.0 * torch.sum(nln_sim_r)
    num_sim_score = torch.sum(nun_sim_r)

    conf_score = (nln_sim_score / num_sim_score).item()  # sim ratio : confidence score
    simratio_score.append(conf_score)
print(simratio_score)
sort_ranking_score, ind_tgt = torch.sort(torch.tensor(simratio_score), descending=True)
print(sort_ranking_score)
print(ind_tgt)