import torch
import torch.nn as nn
import numpy as np
import copy
from ipdb import set_trace


class RefNRIMLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0., no_bn=False):
        super(RefNRIMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.ELU(inplace=True),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_out),
            nn.ELU(inplace=True)
        )
        if no_bn:
            self.bn = None
        else:
            self.bn = nn.BatchNorm1d(n_out)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        orig_shape = inputs.shape
        x = inputs.view(-1, inputs.size(-1))
        x = self.bn(x)
        return x.view(orig_shape)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = self.model(inputs)
        if self.bn is not None:
            return self.batch_norm(x)
        else:
            return x


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, in_c, out_c):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_c, out_c, kernel_size=(1, 1))
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, in_c, in_c)) for i in range(num_layers)])

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_c, out_c, k=3, init_type='random'):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv2d(in_c, out_c, kernel_size=(1, k),
                                      dilation=(1, dilation),
                                      padding=(0, dilation))

        self.relu = nn.ReLU()
        self.conv_1x1 = nn.Conv2d(in_c, out_c, kernel_size=(1, 1))

    def forward(self, x):  # b c 6 t
        out = self.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        return x + out


class VideoModelCoord(nn.Module):
    def __init__(self, opt):
        super(VideoModelCoord, self).__init__()
        self.nr_boxes = 4
        self.nr_actions = opt.num_classes
        self.nr_verb = 61
        self.nr_prep = 59
        self.nr_frames = 16
        self.coord_feature_dim = opt.coord_feature_dim  # 256
        self.longtailclassnum = 30
        dropout = 0.1
        no_bn = False

        self.soft = torch.nn.Softmax(0)

        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim // 2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim // 2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        edges = np.ones(4) - np.eye(4)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()),
                                          requires_grad=True)

        self.mlp1 = RefNRIMLP(self.coord_feature_dim * 2, self.coord_feature_dim, self.coord_feature_dim, dropout,
                              no_bn=no_bn)
        self.mlp2 = RefNRIMLP(self.coord_feature_dim, self.coord_feature_dim, self.coord_feature_dim, dropout,
                              no_bn=no_bn)
        self.mlp3 = RefNRIMLP(self.coord_feature_dim * 2, self.coord_feature_dim, self.coord_feature_dim, dropout,
                              no_bn=no_bn)
        self.mlp4 = RefNRIMLP(self.coord_feature_dim * 2, self.coord_feature_dim, self.coord_feature_dim, dropout,
                              no_bn=no_bn)
        ##############################################################################
        self.num_layers = 4
        self.V_TDconv = SingleStageModel(self.num_layers, self.coord_feature_dim, self.coord_feature_dim)
        self.P_TDconv = SingleStageModel(self.num_layers, self.coord_feature_dim, self.coord_feature_dim)
        self.nohand_TDconv = SingleStageModel(self.num_layers, self.coord_feature_dim, self.coord_feature_dim)
        ###############################################################################
        self.V_TDconv_fusion = nn.Sequential(
            nn.Linear(16 * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.mlp_sub_V1 = nn.Sequential(
            nn.Linear(16 * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim // 2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim // 2),
            nn.ReLU()
        )

        self.mlp_sub_V2 = nn.Sequential(
            nn.Linear(16 * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim // 2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim // 2),
            nn.ReLU()
        )

        self.P_TDconv_fusion = nn.Sequential(
            nn.Linear(16 * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.mlp_sub_P1 = nn.Sequential(
            nn.Linear(16 * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim // 2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim // 2),
            nn.ReLU()
        )

        self.mlp_sub_P2 = nn.Sequential(
            nn.Linear(16 * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim // 2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim // 2),
            nn.ReLU()
        )

        self.nohand_TDconv_fusion = nn.Sequential(
            nn.Linear(16 * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.ReLU()
        )

        self.mlp_sub_nohand1 = nn.Sequential(
            nn.Linear(16 * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim // 2, bias=False),
            nn.ReLU()
        )

        self.mlp_sub_nohand2 = nn.Sequential(
            nn.Linear(16 * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim // 2, bias=False),
            nn.ReLU()
        )

        self.mlp5 = nn.Sequential(
            nn.Linear(2 * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim), nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim), nn.ReLU()
        )

        self.mlp6 = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim), nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim), nn.ReLU()
        )

        self.mlp6_single = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.ReLU()
        )

        self.mlp7 = nn.Sequential(
            nn.Linear(2 * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            nn.ReLU(),
            nn.Linear(self.coord_feature_dim, 512),  # self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_actions)
        )

        self.classifier_v = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_verb)
        )

        self.classifier_p = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_prep)
        )

        if opt.fine_tune:
            self.fine_tune(opt.fine_tune)
        ##############################################################################################
        A_v = np.zeros([12, 12])
        A_p = np.zeros([12, 12])
        for i in range(12):
            for j in range(12):
                # if i != j:
                A_v[i][j] = 1
                A_p[i][j] = 1
        # print('A_v:', A_v)
        # print('A_p:', A_p)
        A_v_norm = self.normalize_digraph(A_v)
        A_p_norm = self.normalize_digraph(A_p)
        # print('A_v_norm:', A_v_norm)
        # print('A_p_norm:', A_p_norm)
        self.Bk_verb = nn.Parameter(torch.from_numpy(A_v_norm.astype(np.float32)))
        self.Bk_prep = nn.Parameter(torch.from_numpy(A_p_norm.astype(np.float32)))
        # print('Bk_verb:', self.Bk_verb)
        # print('Bk_prep:', self.Bk_prep)

        self.conv_vd = nn.Conv1d(self.coord_feature_dim, self.coord_feature_dim, 1)
        self.conv_pd = nn.Conv1d(self.coord_feature_dim, self.coord_feature_dim, 1)

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        h, w = A.shape
        Dn = np.zeros((w, w))
        for i in range(w):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)  # A=h*w，Dn=w*w，dot(A，Dn)=h*w
        return AD

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        # import pdb
        for k, v in weights.items():
            if not 'classifier.4' in k:
                new_weights[k.replace('module.', '')] = v
        # pdb.set_trace()
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'classifier.4' in name:

                param.requires_grad = False
                frozen_weights = frozen_weights + 1

            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def node2edge(self, node_embeddings):
        send_embed = node_embeddings[:, self.send_edges, :]
        recv_embed = node_embeddings[:, self.recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=3)

    def edge2node(self, edge_embeddings):
        if len(edge_embeddings.shape) == 4:  # edge_embeddings [N, V(V-1), T, Dc=128]
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)  # old_shape[1] = 12
            incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(old_shape[0], -1, old_shape[2],
                                                                             old_shape[3])
        else:
            incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming / (self.nr_boxes - 1)  # TODO: do we want this average?

    def forward(self, box_input, video_label, flag, isval):
        box_input = box_input.cuda()
        b = box_input.shape[0]

        N, E, T, V = b, 12, 16, 4
        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(b * self.nr_boxes * self.nr_frames, 4)
        bf = self.coord_to_feature(box_input)  # [N*V*T,128]
        bf = bf.view(b, self.nr_boxes, self.nr_frames, self.coord_feature_dim)  # [N, V, T, D]

        x = self.node2edge(bf).reshape(N * E * T, -1)  # [N, V(V-1), T, D'=256]
        x = self.mlp1(x)  # [N*V(V-1)*T, D]
        x_skip = x
        x = x.reshape(N, E, T, -1)
        x = self.edge2node(x)  # [N, V, T, D]
        x = x.reshape(N * V * T, -1)
        x = self.mlp2(x)  # [N, V, T, D]
        x = x.reshape(N, V, T, -1)

        x = self.node2edge(x).reshape(N * E * T, -1)  # 256
        x = self.mlp3(x)
        x = torch.cat((x, x_skip), dim=-1)  # [N, E, T, D'=256]
        x = self.mlp4(x).reshape(N, E, T, -1).transpose(0, 1).reshape(E, -1)  # [N, E, T, D]

        edge_feature = x.tolist()

        Dim = self.coord_feature_dim
        zeros = [0] * N * T * Dim
        edge_feature.insert(0, zeros)
        edge_feature.insert(5, zeros)
        edge_feature.insert(10, zeros)
        edge_feature.insert(15, zeros)

        edge_feature = torch.tensor(edge_feature).cuda()  # [16, xxx]
        edge_feature = edge_feature.reshape(4, 4, N, T, Dim).permute(2, 0, 1, 3, 4).reshape(N * 4, -1)  # [N*4, 4*T*D]

        edge_verb = []
        edge_prep = []
        id_global = []
        index_global = []

        for i in range(b):
            edge_verb.append(edge_feature[i * 4:(i * 4 + 4)])
            edge_prep.append(edge_feature[i * 4:(i * 4 + 4)])
            index_global.append(i)
            id_global.append(i)

        index_global = torch.tensor(index_global).cuda()

        if len(edge_verb) != 0:
            # [N*4, 4*T*Dim]
            edge_verb = torch.stack([edge_verb[i] for i in range(len(edge_verb))]).reshape(-1, 4 * T * Dim).cuda()
            N_verb = int((edge_verb.shape[0]) / 4)
        else:
            edge_verb = None
            N_verb = 0

        if len(edge_prep) != 0:
            # [N*4, 4*T*Dim]
            edge_prep = torch.stack([edge_prep[i] for i in range(len(edge_prep))]).reshape(-1, 4 * T * Dim).cuda()
            N_prep = int((edge_prep.shape[0]) / 4)
        else:
            edge_prep = None
            N_prep = 0

        if N_verb != 0:

            # dir verb subgraph
            verb_graph_edge = edge_verb  # prep_graph_edge_t
            verb_graph_edge = verb_graph_edge.reshape(N_verb, 4, 4, -1)  # [N_perb, 4, 4, T*Dim]
            verb_e = []
            for n in range(N_verb):
                verb_e.append([])
            for n in range(N_verb):
                for i in range(4):
                    for j in range(4):
                        if not verb_graph_edge[n, i, j].equal(torch.zeros(T * Dim).cuda()):
                            verb_e[n].append(verb_graph_edge[n, i, j])
            for n in range(N_verb):
                verb_e[n] = torch.stack([verb_e[n][i] for i in range(len(verb_e[n]))])
            verb_final = torch.stack([verb_e[n] for n in range(len(verb_e))]).cuda()  # [N_verb, 12, T*Dim]

            # prep subgraph
            prep_graph_edge = edge_prep  # prep_graph_edge_t

            prep_graph_edge = prep_graph_edge.reshape(N_prep, 4, 4, -1)  # [N_verb, 4, 4, T*Dim]

            prep_e = []

            for n in range(N_prep):
                prep_e.append([])
            for n in range(N_prep):
                for i in range(4):
                    for j in range(4):
                        if not prep_graph_edge[n, i, j].equal(torch.zeros(T * Dim).cuda()):
                            prep_e[n].append(prep_graph_edge[n, i, j])

            for n in range(N_prep):
                prep_e[n] = torch.stack([prep_e[n][i] for i in range(len(prep_e[n]))])
            prep_final = torch.stack([prep_e[n] for n in range(len(prep_e))]).cuda()

            torch.backends.cudnn.enabled = False

            verb_final_conv = verb_final.reshape(N_verb, 12, T, Dim).permute(0, 3, 1, 2)  # [10,6,T*Dim] to [10,Dim,6,T]
            prep_final_conv = prep_final.reshape(N_verb, 12, T, Dim).permute(0, 3, 1, 2)
            verb_final_conv = self.V_TDconv(verb_final_conv).reshape(N_verb * 12, T * Dim)  # [10,Dim,6,T] to [10*6,T*Dim]
            prep_final_conv = self.P_TDconv(prep_final_conv).reshape(N_verb * 12, T * Dim)

            torch.autograd.set_detect_anomaly(True)

            verb_final = self.V_TDconv_fusion(verb_final_conv).reshape(N_verb, 12, Dim)  # [10*6,16*256] to [10,6,256]
            prep_final = self.P_TDconv_fusion(prep_final_conv).reshape(N_verb, 12, Dim)

            A1_v = self.Bk_verb
            A2_v = verb_final.permute(0, 2, 1)  # N,C,V
            verb_final_temp = self.conv_vd(torch.matmul(A2_v, A1_v)).permute(0, 2, 1)  # N,C,V to N,V,C [12,6,256]

            verb_final = verb_final + 0.5 * verb_final_temp

            A1_p = self.Bk_prep
            A2_p = prep_final.permute(0, 2, 1)  # N,C,V
            prep_final_temp = self.conv_pd(torch.matmul(A2_p, A1_p)).permute(0, 2, 1)  # N,C,V to N,V,C
            prep_final = prep_final + 0.5 * prep_final_temp

            verb_final = torch.mean(verb_final, dim=1)  # (10, 12, 256) to (10, 256)
            prep_final = torch.mean(prep_final, dim=1)  # (10, 12, 256) to (10, 256)

            # print('Bk_verb', self.Bk_verb)
            # print('Bk_prep', self.Bk_prep)

            box_features = torch.cat((verb_final, prep_final), dim=-1)  # [N_verb, 512]
            box_features = self.mlp5(box_features)  # [N_verb, 256]
            cls_hand = self.classifier(box_features)  # [N_verb, 174]

            if len(cls_hand.shape) == 1:
                cls_hand = cls_hand.unsqueeze(0)

            video_label_hand = []
            for id in index_global:
                video_label_hand.append(video_label[id])
            video_label_hand = torch.stack([video_label_hand[i] for i in range(len(video_label_hand))])

        else:
            cls_hand = None
            video_label_hand = None

        # TODO:恢复index
        cls = torch.zeros(N, self.nr_actions).cuda()

        label = torch.zeros(N, 1).cuda()
        if cls_hand is not None:
            for i in range(index_global.shape[0]):
                cls[index_global[i]] = cls_hand[i]
                label[index_global[i]] = video_label_hand[i]
        label = label.squeeze()

        return cls, label.long(), flag
