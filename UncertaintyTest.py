with torch.no_grad():
    # for batch_idx, ((data, _), target, _) in enumerate(tqdm(train_loader)):
    for batch_idx, ((_, data_t_og, data_t_raug), label_t, indices_t) in enumerate(tqdm(self.tgt_loader)):
        data_t_og = data_t_raug.to(device)

def get_Uncertainty(fc2_s, fc2_s1, fc2_s2, fc2_s3, device):

    fc2_s = nn.Softmax(-1)(fc2_s).to(device)
    fc2_s1 = nn.Softmax(-1)(fc2_s1).to(device)
    fc2_s2 = nn.Softmax(-1)(fc2_s2).to(device)
    fc2_s3 = nn.Softmax(-1)(fc2_s3).to(device)

    fc2_s = torch.unsqueeze(fc2_s, 1).to(device)
    fc2_s1 = torch.unsqueeze(fc2_s1, 1).to(device)
    fc2_s2 = torch.unsqueeze(fc2_s2, 1).to(device)
    fc2_s3 = torch.unsqueeze(fc2_s3, 1).to(device)


    c = torch.cat((fc2_s, fc2_s1, fc2_s2, fc2_s3), dim=1).to(device)
    d = torch.std(c, 1).to(device)

    uncertainty = torch.mean(d,1).to(device)

    return uncertainty