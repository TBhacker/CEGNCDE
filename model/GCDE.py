import torch
import torch.nn.functional as F
import torch.nn as nn
import controldiffeq
import math
from vector_fields import *

class NeuralGCDE(nn.Module):
    def __init__(self, args, func_f, func_g, input_channels, hidden_channels, output_channels, initial, device, atol, rtol, solver):
        super(NeuralGCDE, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = input_channels
        self.hidden_dim = hidden_channels
        self.output_dim = output_channels
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.device = device
          
        self.default_graph = args.default_graph
        #self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        
        self.func_f = func_f
        self.func_g = func_g
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        #self.embedding = nn.Linear(1, input_channels)
        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        # self.end_liner = torch.nn.Linear(self.hidden_dim, self.output_dim)
        # self.end_conv1 = nn.Conv2d(
        #     in_channels=args.horizon, out_channels=args.horizon, kernel_size=1, bias=True,
        # )
        # self.end_conv2 = nn.Conv2d(
        #     in_channels=self.hidden_dim, out_channels=self.output_dim, kernel_size=1, bias=True,
        # )

        self.steps_per_day =288
        self.tod_embedding_dim=16
        self.dow_embedding_dim=16
        self.input_embedding_dim = 32
        self.adaptive_embedding_dim=32
        self.in_steps=12,
        self.input_proj = nn.Linear(1, self.input_embedding_dim)
        self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        self.adaptive_embedding  = nn.Embedding(7, self.dow_embedding_dim)
        # self.adaptive_embedding = nn.init.xavier_uniform_(
        #         nn.Parameter(torch.empty(in_steps,self.num_node, adaptive_embedding_dim))
        #     )
    
        
        self.init_type = 'fc'
        if self.init_type == 'fc':
            self.initial_h = torch.nn.Linear(4, self.hidden_dim)
            self.initial_z = torch.nn.Linear(4, self.hidden_dim)
        elif self.init_type == 'conv':
            self.start_conv_h = nn.Conv2d(in_channels=input_channels,
                                            out_channels=hidden_channels,
                                            kernel_size=(1,1))
            self.start_conv_z = nn.Conv2d(in_channels=input_channels,
                                            out_channels=hidden_channels,
                                            kernel_size=(1,1))
            


        # computing the positional encodings once in log space
        # pe = torch.zeros(12, self.input_dim)
        # for pos in range(12):
        #     for i in range(0, self.input_dim, 2):
        #         pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.input_dim)))
        #         pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.input_dim)))

        # self.pe = pe.unsqueeze(0).unsqueeze(0).permute(0,2,1,3).to(device)  # (1, 1, T_max, d_model)

    def forward(self, times, coeffs):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        #coeffs = self.enc_embed_layer(coeffs)
        
        # x = x[0]
        # time = torch.linspace(0, 11, 12).to(self.device)
        # tod = x[...,1]
        # dow = x[...,2]
        # x = self.input_proj(x)   
        # # features = [x]
        # # tod_emb = self.tod_embedding(
        # #         (tod * self.steps_per_day).long()
        # #     )
        # # features.append(tod_emb)
        
        # # dow_emb = self.dow_embedding(
        # #         dow.long()
        # #     )
        # # features.append(dow_emb)
        # # x = torch.cat(features, dim=-1)

        # coeffs = controldiffeq.natural_cubic_spline_coeffs(time, x.transpose(1,2))
        spline = controldiffeq.NaturalCubicSpline(times, coeffs)
       
        if self.init_type == 'fc':
            x0 = spline.evaluate(times[0])
            h0 = self.initial_h(x0)
            z0 = self.initial_z(spline.evaluate(times[0]))
        elif self.init_type == 'conv':
            h0 = self.start_conv_h(spline.evaluate(times[0]).transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()
            z0 = self.start_conv_z(spline.evaluate(times[0]).transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()

        z_t = controldiffeq.cdeint_gde_dev_test(dX_dt=spline.derivative, #dh_dt
                                   h0=h0,
                                   z0=z0,
                                   func_f=self.func_f,
                                   func_g=self.func_g,
                                   input_proj=self.input_proj,
                                   tod_embedding = self.tod_embedding,
                                   dow_embedding = self.dow_embedding,
                                   adaptive_embedding = self.adaptive_embedding,
                                   t=times,
                                   method=self.solver,
                                   atol=self.atol,
                                   rtol=self.rtol)

        # init_state = self.encoder.init_hidden(source.shape[0])
        # output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        z_T = z_t[-1:,...].transpose(0,1)  #B 1 N C
        #z_T = z_t.transpose(0,1) #B 12 N C
       
        #CNN based predictor
        output = self.end_conv(z_T)                         #B, T*C, N, 1
        #output = self.end_liner(z_T) 
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C
        # skip = z_T.permute(0, 3, 2, 1)
        # skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        # skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))
        return output

        return output





