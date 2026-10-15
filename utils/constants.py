import os

import torch
device = torch.device('cuda:{}'.format(3) if torch.cuda.is_available() else 'cpu')

PAD = 0
UNK = 2
BOS = 3
EOS = 1

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
time_step_split = 5
n_heads = 14
step_len = 5
scale_time_loss = 100
MAX_CAS_LEN = 200
MAX_RAW_CAS_LEN = 500

CAS_ID_INIT = 900000000


DATA_ROOT = os.environ.get('STGSM_DATA_ROOT', '../data/')
if not DATA_ROOT.endswith('/'):
    DATA_ROOT += '/'


class Options(object):
    def __init__(self, data_name='twitter'):


        self.train_data = DATA_ROOT + data_name + '/cascade.txt'

        self.valid_data = DATA_ROOT + data_name + '/cascadevalid.txt'

        self.test_data = DATA_ROOT + data_name + '/cascadetest.txt'


        self.train_data_id = DATA_ROOT + data_name + '/cascade_id.txt'

        self.valid_data_id = DATA_ROOT + data_name + '/cascadevalid_id.txt'

        self.test_data_id = DATA_ROOT + data_name + '/cascadetest_id.txt'


        self.u2idx_dict = DATA_ROOT + data_name + '/u2idx.pickle'
        self.idx2u_dict = DATA_ROOT + data_name + '/idx2u.pickle'


        self.ui2idx_dict = DATA_ROOT + data_name + '/ui2idx.pickle'
        self.idx2ui_dict = DATA_ROOT + data_name + '/idx2ui.pickle'


        self.t2idx_dict = DATA_ROOT + data_name+ '/t2idx.pickle'
        self.idx2t_dict = DATA_ROOT + data_name+ '/idx2t.pickle'


        self.save_path = ''


        self.net_data = DATA_ROOT + data_name + '/edges.txt'
        self.net_data_refined = DATA_ROOT + data_name + '/edges_refined.txt'


        self.item_net_data = DATA_ROOT + data_name + "/edges_item.txt"


        self.repost_net_data = DATA_ROOT + data_name + "/edges_reposts.txt"
        self.repost_net_data1 = DATA_ROOT + data_name + "/output_network.txt"


        self.repost_time_data = DATA_ROOT + data_name + "/edges_reposts_time.txt"


        self.item_net_data_with_time_weight = DATA_ROOT + data_name + "/edges_item_weights.txt"
        self.repost_net_data_with_time_weight = DATA_ROOT + data_name + "/edges_reposts_weights.txt"


        self.item_net_data_trimmed = DATA_ROOT + data_name + "/edges_item_trimmed.txt"
        self.item_net_data_with_time_weight_trimmed = DATA_ROOT + data_name + "/edges_item_trimmed_weights.txt"

        self.repost_net_data_trimmed = DATA_ROOT + data_name + "/edges_reposts_trimmed.txt"
        self.repost_net_data_with_time_weight_trimmed = DATA_ROOT + data_name + "/edges_reposts_trimmed_weights.txt"


        self.item_inner_net_data=DATA_ROOT + data_name+"/edges_item_inner.txt"
        self.item_inner_net_data_refined=DATA_ROOT + data_name+"/edges_item_inner_refined.txt"


        self.user_repost_file = DATA_ROOT + data_name+"/user_repost_list.pickle"


        self.user_hidden_net = DATA_ROOT + data_name+'/edges_user_hidden.txt'


        self.batch_size = 16
        self.embed_dim = 64


class Logger(object):
    def __init__(self, filename="Default.log"):
        import sys,os
        self.terminal = sys.stdout
        if os.path.exists("./log") is False:
            os.mkdir("./log")
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
