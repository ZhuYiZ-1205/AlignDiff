import torch.utils.data as data_utils
import torch


class TrainDataset(data_utils.Dataset):
    def __init__(self, id2seq, max_len):
        self.id2seq = id2seq
        self.max_len = max_len

    def __len__(self):
        return len(self.id2seq)  # 131413

    def __getitem__(self, index):
        seq = self._getseq(index) 
        labels = [seq[-1]] 
        tokens = seq[:-1] 
        targets = seq[1:]
        tokens = tokens[-self.max_len:] 
        mask_len = self.max_len - len(tokens)
        seq_len = [len(tokens)]
        tokens = [0] * mask_len + tokens 
        targets = targets[-self.max_len:]
        target_mask_len = self.max_len - len(targets)
        targets = [0] * target_mask_len + targets
        # print(f"Returning tokens: {tokens}, labels: {labels}, targets: {targets}")
        return torch.LongTensor(tokens), torch.LongTensor(labels), torch.LongTensor(targets), torch.LongTensor(seq_len) 

    def _getseq(self, idx):
        return self.id2seq[idx]


class Data_Train():
    def __init__(self, data_train, args):
        self.u2seq = data_train
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.split_onebyone()

    def split_onebyone(self):
        self.id_seq = {} 
        self.id_seq_user = {}  
        idx = 0  
        for user_temp, seq_temp in self.u2seq.items(): 
            for star in range(len(seq_temp) - 1):  
                self.id_seq[idx] = seq_temp[:star + 2]
                self.id_seq_user[idx] = user_temp
                idx += 1

    def get_pytorch_dataloaders(self):
        dataset = TrainDataset(self.id_seq, self.max_len)  
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)


class ValDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        targets = self.u2seq[user][1:] + self.u2answer[user]
        targets = targets[-self.max_len:]
        targets_padding_len = self.max_len - len(targets)
        targets = [0] * targets_padding_len + targets
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq_len = [len(seq)]
        seq = [0] * padding_len + seq
        return torch.LongTensor(seq), torch.LongTensor(answer), torch.LongTensor(targets), torch.LongTensor(seq_len)


class Data_Val():
    def __init__(self, data_train, data_val, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2answer = data_val  
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = ValDataset(self.u2seq, self.u2answer, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader


class TestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2_seq_add, u2answer, max_len):
        self.u2seq = u2seq
        self.u2seq_add = u2_seq_add
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user][1:] + self.u2seq_add[user]
        # seq = self.u2seq[user]
        answer = self.u2answer[user]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq_len = [len(seq)]
        seq = [0] * padding_len + seq

        targets = self.u2seq[user] + self.u2seq_add[user] + self.u2answer[user]
        targets = targets[-self.max_len:]
        targets_padding_len = self.max_len - len(targets)
        targets = [0] * targets_padding_len + targets
        return torch.LongTensor(seq), torch.LongTensor(answer), torch.LongTensor(targets), torch.LongTensor(seq_len)


class Data_Test():
    def __init__(self, data_train, data_val, data_test, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2seq_add = data_val
        self.u2answer = data_test
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = TestDataset(self.u2seq, self.u2seq_add, self.u2answer, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader

