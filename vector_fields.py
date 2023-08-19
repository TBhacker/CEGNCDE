import torch
import torch.nn as nn
import torch.nn.functional as F
from AGCRN.AGCN import AVWGCN
# ffff
class FinalTanh_f(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_f, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        
        self.linears = nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #64,307，128  -> # 64,307，128*2

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        #64 307 128
        z = self.linear_in(z)
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        # z: torch.Size([64, 307, 128])
        # self.linear_out(z): torch.Size([64, 307, 128*2])
        
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels) 
        # torch.Size([64, 307, 128，2])   
        z = z.tanh()
        # torch.Size([64, 307, 128，2])
        return z

class FinalTanh_f_prime(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_f_prime, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        
        self.linears = nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        # self.linear_out = nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        self.linear_out = nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4 

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        # z: torch.Size([64, 207, 32])
        # self.linear_out(z): torch.Size([64, 207, 64])
        # z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)    
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)    
        z = z.tanh()
        return z

class FinalTanh_f2(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_f2, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.start_conv = torch.nn.Conv2d(in_channels=hidden_channels,
                                    out_channels=hidden_channels,
                                    kernel_size=(1,1))

        self.linears = torch.nn.ModuleList(torch.nn.Conv2d(in_channels=hidden_channels,
                                    out_channels=hidden_channels,
                                    kernel_size=(1,1))
                                           for _ in range(num_hidden_layers - 1))
        
        self.linear_out = torch.nn.Conv2d(in_channels=hidden_channels,
                                    out_channels=input_channels*hidden_channels,
                                    kernel_size=(1,1))

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        # z: torch.Size([64, 207, 32])
        z = self.start_conv(z.transpose(1,2).unsqueeze(-1))
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()

        z = self.linear_out(z).squeeze().transpose(1,2).view(*z.transpose(1,2).shape[:-2], self.hidden_channels, self.input_channels)
        z = z.tanh()
        return z

class VectorField_g_gcn_init(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type):
        super(VectorField_g_gcn, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        #FIXME:
        
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels)  
        
        self.g_type = g_type
        if self.g_type == 'agc':
            #self.node_embeddings = nn.Parameter(torch.randn(num_nodes, hidden_channels), requires_grad=True)
            self.nodes_pool = nn.Parameter(torch.randn(hidden_channels, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor( embed_dim, cheb_k,hidden_hidden_channels, hidden_hidden_channels), requires_grad=True)
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim,hidden_hidden_channels), requires_grad=True)


    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)
    def forward(self, z, h):
        """
        
        - z:64 307 128
        - h:64 307 128
        """
        
        z = self.linear_in(z)
        z = z.relu()

        if self.g_type == 'agc':
            z = self.agc(z,h)
        else:
            raise ValueError('Check g_type argument')

        #FIXME:
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
        z = z.tanh()
        return z #torch.Size([64, 307, 128, 128])
    
    def agc(self, z):

        """
        Adaptive Graph Convolution
        - Node Adaptive Parameter Learning
        - Data Adaptive Graph Generation
        
        """
        
        node_num = self.node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        
        laplacian=False
        if laplacian == True:
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            support_set = [supports, -torch.eye(node_num).to(supports.device)]
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            # support_set = [-supports]
        else:
            support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        # supports：2  307 307 
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out 307 2 128 128
        bias = torch.matmul(self.node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, z)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        z = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return z
    
class VectorField_g_gcn(torch.nn.Module):
    def __init__(self, coefficient, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type):
        super(VectorField_g_gcn, self).__init__()
        self.coefficient = coefficient
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        #FIXME:
        
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels)  
        
        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))
    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)
    def forward(self, z, h):
        """
        
        - z:64 307 128
        - h:64 307 128
        """
        
        z = self.linear_in(z)
        z = z.relu()

        if self.g_type == 'agc':
            z = self.agc(z,h)
        else:
            raise ValueError('Check g_type argument')

        #FIXME:
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
        z = z.tanh()
        return z #torch.Size([64, 307, 128, 128])

    def agc(self, z, h):
        """
        Adaptive Graph Convolution
        - Node Adaptive Parameter Learning
        - Data Adaptive Graph Generation
        """
        h = h[:,:,:,0]
        # 64 307 128

        node_num = h.shape[1]
        supports_embedding = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)

        supports = self.coefficient*F.softmax(F.relu(torch.matmul(h, h.transpose(-2, -1))), dim=2) +(1-self.coefficient)*supports_embedding
        # laplacian=False
        laplacian=False
        supports_eye = torch.eye(node_num).unsqueeze(0).repeat(supports.shape[0],1,1).to(supports.device)
        if laplacian == True:
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            support_set = [supports, -supports_eye]
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            # support_set = [-supports]
        else:
            support_set = [supports_eye, supports]
        #default cheb_k = 2
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])

        z_g = []
        for support in support_set:
            z_g.append(torch.matmul(support, z) )
        #list 2[64 307 128 ]
        z_g = torch.stack(z_g, dim=0)
        #2 64 307 128 
        # supports = torch.stack(support_set, dim=0)
        # # 2 64 307 307
        z_g = z_g.permute(1,2,0,3)
        # # B, N, cheb_k, dim_in 64 307 2 128
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  
        #N, cheb_k, dim_in, dim_out 307 2 128 128
        bias = torch.matmul(self.node_embeddings, self.bias_pool)                       
        #N, dim_out 307 128
        z_g = torch.einsum('bnki,nkio->bno', z_g, weights) + bias 
       
        return z_g
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class VectorField_g_former(torch.nn.Module):
    def __init__(self, coefficient, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type):
        super(VectorField_g_former, self).__init__()
        self.coefficient = coefficient
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.geo_mask = torch.load("./data/PEMS08/geo_mask.pth")
        self.sem_mask = torch.load("./data/PEMS08/sem_mask.pth")

      

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        #FIXME:
        
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels)  
        
        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)

            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))
        if self.g_type == 'att_cheb':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))
            self.geo_q_w = nn.Parameter(torch.randn(hidden_channels, hidden_channels), requires_grad=True)
            self.geo_k_w = nn.Parameter(torch.randn(hidden_channels, hidden_channels), requires_grad=True)
            self.geo_v_w = nn.Parameter(torch.randn(hidden_channels, hidden_channels), requires_grad=True)
            self.norm1 = nn.LayerNorm(hidden_channels)
            self.norm2 = nn.LayerNorm(hidden_channels)
            self.porj_drop = nn.Dropout(0.1)
            self.mlp = Mlp(in_features=hidden_channels, hidden_features=hidden_channels, act_layer=nn.GELU, drop=0.)
        if self.g_type == 'GRU':
            
            
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            
            self.geo_num_heads = 1
            self.sem_num_heads = 1
            self.head_dim = hidden_channels // (self.geo_num_heads + self.sem_num_heads)
        
            self.scale = self.head_dim ** -0.5
            self.geo_ratio = self.geo_num_heads / (self.geo_num_heads + self.sem_num_heads)
            self.sem_ratio = self.sem_num_heads / (self.geo_num_heads + self.sem_num_heads)

            self.geo_q_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            self.geo_k_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            self.geo_v_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            self.geo_att_drop = nn.Dropout(0.1)
            self.sem_q_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            self.sem_k_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            self.sem_v_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            self.gate = AVWGCN(int(hidden_channels*self.geo_ratio), 2*int(hidden_channels*self.geo_ratio), cheb_k, embed_dim)
            self.update = AVWGCN(int(hidden_channels*self.geo_ratio), int(hidden_channels*self.geo_ratio), cheb_k, embed_dim)
            self.norm1 = nn.LayerNorm(hidden_channels)
            self.norm2 = nn.LayerNorm(hidden_channels)
            self.porj_drop = nn.Dropout(0.1)
            self.mlp = Mlp(in_features=hidden_channels, hidden_features=hidden_channels, act_layer=nn.GELU, drop=0.)
        if self.g_type == 'former':
            self.geo_num_heads = 2
            self.sem_num_heads = 2
            self.head_dim = hidden_channels // (self.geo_num_heads + self.sem_num_heads)
        
            self.scale = self.head_dim ** -0.5
            self.geo_ratio = self.geo_num_heads / (self.geo_num_heads + self.sem_num_heads)
            self.sem_ratio = self.sem_num_heads / (self.geo_num_heads + self.sem_num_heads)
            self.enbeddings_ratio = 1 - self.geo_ratio - self.sem_ratio
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))
            self.geo_q_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            self.geo_k_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            self.geo_v_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            # self.geo_q_w = nn.Parameter(torch.randn(hidden_channels, int(hidden_channels*self.geo_ratio)), requires_grad=True)
            # self.geo_k_w = nn.Parameter(torch.randn(hidden_channels, int(hidden_channels*self.geo_ratio)), requires_grad=True)
            # self.geo_v_w = nn.Parameter(torch.randn(hidden_channels, int(hidden_channels*self.geo_ratio)), requires_grad=True)
           
            self.geo_att_drop = nn.Dropout(0.1)
            self.sem_q_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            self.sem_k_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            self.sem_v_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            # self.sem_q_w = nn.Parameter(torch.randn(hidden_channels, int(hidden_channels*self.geo_ratio)), requires_grad=True)
            # self.sem_k_w = nn.Parameter(torch.randn(hidden_channels, int(hidden_channels*self.geo_ratio)), requires_grad=True)
            # self.sem_v_w = nn.Parameter(torch.randn(hidden_channels, int(hidden_channels*self.geo_ratio)), requires_grad=True)
           
            self.sem_att_drop = nn.Dropout(0.1)
            self.proj = nn.Linear(hidden_channels, hidden_channels)
            self.norm1 = nn.LayerNorm(hidden_channels)
            self.norm2 = nn.LayerNorm(hidden_channels)
            self.porj_drop = nn.Dropout(0.1)
            self.mlp = Mlp(in_features=hidden_channels, hidden_features=hidden_channels, act_layer=nn.GELU, drop=0.)

        if self.g_type == 'former_GRU':
            ##测试一下多头的作用 ,结果为1是更好
            self.geo_num_heads = 1
            self.sem_num_heads = 1
            self.head_dim = hidden_channels // (self.geo_num_heads + self.sem_num_heads)
         #   hidden_channels 64 self.head_dim 8
            self.scale = self.head_dim ** -0.5
            self.geo_ratio = self.geo_num_heads / (self.geo_num_heads + self.sem_num_heads)
            self.sem_ratio = self.sem_num_heads / (self.geo_num_heads + self.sem_num_heads)
            self.enbeddings_ratio = 1 - self.geo_ratio - self.sem_ratio
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k

            self.gate = AVWGCN(hidden_channels, 2*hidden_channels, cheb_k, embed_dim)
            self.update = AVWGCN(hidden_channels, hidden_channels, cheb_k, embed_dim)
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))
            self.geo_q_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            self.geo_k_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            self.geo_v_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            self.geo_att_drop = nn.Dropout(0.1)
            self.sem_q_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            self.sem_k_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            self.sem_v_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
            self.sem_att_drop = nn.Dropout(0.1)
            self.proj = nn.Linear(hidden_channels, hidden_channels)
            self.norm1 = nn.LayerNorm(hidden_channels)
            self.norm2 = nn.LayerNorm(hidden_channels)
            self.porj_drop = nn.Dropout(0.1)
            self.mlp = Mlp(in_features=hidden_channels, hidden_features=hidden_channels, act_layer=nn.GELU, drop=0.)
        if self.g_type == 'pdformer':
            self.encoder_block = STEncoderBlock(hidden_channels=hidden_channels,geo_num_heads=4,sem_num_heads=4,nums_node=num_nodes,
                            embed_dim=embed_dim,cheb_k=cheb_k)
            # self.encoder_blocks = nn.ModuleList([
            #     STEncoderBlock(hidden_channels=hidden_channels,geo_num_heads=4,sem_num_heads=4,nums_node=num_nodes,
            #                embed_dim=embed_dim,cheb_k=cheb_k) for i in range(1)])
            # self.skip_convs = nn.ModuleList([
            #     nn.Linear(hidden_channels, hidden_channels) for _ in range(1)])
            # self.liner1 = nn.Linear(hidden_channels, hidden_channels)
            # self.liner2 = nn.Linear(hidden_channels, hidden_channels)

        if self.g_type == 'self_att_chebnet':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))
            ##self attetion
            self.scale = hidden_channels ** -0.5
            self.geo_q_w = nn.Linear(hidden_channels, hidden_channels)
            self.sem_q_w = nn.Linear(hidden_channels, hidden_channels)
            self.geo_k_w = nn.Linear(hidden_channels, hidden_channels)
            self.sem_k_w = nn.Linear(hidden_channels, hidden_channels)

            self.proj = nn.Linear(hidden_channels, hidden_channels)
            self.norm1 = nn.LayerNorm(hidden_channels)
            self.norm2 = nn.LayerNorm(hidden_channels)
            self.mlp = Mlp(in_features=hidden_channels, hidden_features=hidden_channels, act_layer=nn.GELU, drop=0.)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)
    def forward(self, z, h):
        """
        
        - z:64 307 128
        - h:64 307 128
        """
        h = h[:,:,:,0]
        z = self.linear_in(z)
        z = z.relu()

        if self.g_type == 'agc':
            z = self.agc(z,h)
        elif self.g_type == 'att_cheb': 
            z = z+self.att_cheb(z,self.norm1(h))
            z = z + self.mlp(self.norm2(z))
        elif self.g_type == 'GRU': 
            z = z+ self.GRU(z, self.norm1(h))
            z = z + self.mlp(self.norm2(z))
       
        elif self.g_type == 'former': 
            z = z + self.former(z,self.norm1(h))
            z = z + self.mlp(self.norm2(z))
        ##############################################
        elif self.g_type == 'former_GRU': 
            z = z + self.former_GRU(z,self.norm1(h))
            z = z + self.mlp(self.norm2(z))
        ##############################################
        elif self.g_type == 'pdformer': 
            
            z = self.encoder_block(z,h)

        elif self.g_type == 'self_att_chebnet': 
            z = z+ self.self_att_chebnet(self.norm1(z))
            z = z + self.mlp(self.norm2(z))
        else:
            raise ValueError('Check g_type argument')

        #FIXME:
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
        z = z.tanh()
        return z #torch.Size([64, 307, 128, 128])

    def agc(self, z, h):

        """
        Adaptive Graph Convolution
        - Node Adaptive Parameter Learning
        - Data Adaptive Graph Generation
        """
        #由h生成动态矩阵，维度为64 307 307，切比雪夫K=2 
        h = h[:,:,:,0]
        # 64 307 128



       

        node_num = h.shape[1]
        #静态
        supports_embedding = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        #静态和动态结合
        supports = self.coefficient*F.softmax(F.relu(torch.matmul(h, h.transpose(-2, -1))), dim=2) +(1-self.coefficient)*supports_embedding
        # laplacian=False
        laplacian=False
        supports_eye = torch.eye(node_num).unsqueeze(0).repeat(supports.shape[0],1,1).to(supports.device)
        if laplacian == True:
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            support_set = [supports, -supports_eye]
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            # support_set = [-supports]
        else:
            support_set = [supports_eye, supports]
        #default cheb_k = 2
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])

        z_g = []
        for support in support_set:
            z_g.append(torch.matmul(support, z) )
        #list 2[64 307 128 ]
        z_g = torch.stack(z_g, dim=0)
        #2 64 307 128 
        # supports = torch.stack(support_set, dim=0)
        # # 2 64 307 307
        z_g = z_g.permute(1,2,0,3)
        # # B, N, cheb_k, dim_in 64 307 2 128
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  
        #N, cheb_k, dim_in, dim_out 307 2 128 128
        bias = torch.matmul(self.node_embeddings, self.bias_pool)                       
        #N, dim_out 307 128
        z_g = torch.einsum('bnki,nkio->bno', z_g, weights) + bias 
       
        return z_g

    def att_cheb(self, z,h):
        #z 64 170 64 B N I
        #####动态图和静态图融合后 mask，做chebnet
        #ngeo_num_heads = 4
        geo_q = torch.einsum('bni,io->bno', h, self.geo_q_w) 
        geo_k = torch.einsum('bni,io->bno', h, self.geo_k_w)
        geo_v = z
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)).softmax(dim=-1)
        #加上静态
        supports_embedding = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        #静态和动态结合
        #supports = self.coefficient*geo_attn +(1-self.coefficient)*supports_embedding
        #将两个图按位乘
        supports = torch.mul(geo_attn,supports_embedding)
        supports.masked_fill_(self.geo_mask.to(z.device), float('-inf'))
        supports = supports.softmax(dim=-1)
        # laplacian=False
        laplacian=False
        
        node_num = geo_v.shape[1]
        supports_eye = torch.eye(node_num).unsqueeze(0).repeat(supports.shape[0],1,1).to(supports.device)
        if laplacian == True:
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            support_set = [supports, -supports_eye]
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            # support_set = [-supports]
        else:
            support_set = [supports_eye, supports]
        #default cheb_k = 2
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])

        z_g = []
        for support in support_set:
            z_g.append(torch.matmul(support, geo_v) )
        #list 2[64 307 128 ]
        z_g = torch.stack(z_g, dim=0)
        #2 64 307 128 
        # supports = torch.stack(support_set, dim=0)
        # # 2 64 307 307
        z_g = z_g.permute(1,2,0,3)
        # # B, N, cheb_k, dim_in 64 307 2 128
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  
        #N, cheb_k, dim_in, dim_out 307 2 128 128
        bias = torch.matmul(self.node_embeddings, self.bias_pool)                       
        #N, dim_out 307 128
        z_g = torch.einsum('bnki,nkio->bno', z_g, weights) + bias 
       
        return z_g
    def AGCRN(self, x, state, node_embeddings,supports):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        
        #input_and_state = torch.cat((x, state), dim=-1)
        #z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, supports))
        z_r = torch.sigmoid(self.gate(state, node_embeddings, supports))
        z, r = torch.split(z_r, int(self.hidden_channels*self.geo_ratio), dim=-1)
        #candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(z*state, node_embeddings, supports))
        h = r*state + (1-r)*hc
        return h
    def GRU(self, z, h):
        #z 64 170 64 B N I
        #####动态图和静态图融合后 mask，做chebnet
        #ngeo_num_heads = 4
        A_static = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        geo_q = self.geo_q_w(h)
        geo_k = self.geo_k_w(h)
        #z = torch.cat([z, h], dim=-1)
        geo_v = self.geo_v_w(h)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)).softmax(dim=-1)
        Adj_geo = self.coefficient*geo_attn +(1-self.coefficient)*A_static# 64 170 17
        Adj_geo.masked_fill_(self.geo_mask.to(z.device), float('-inf'))
        Adj_geo = Adj_geo.softmax(dim=-1)#64 170 170

        sem_q = self.sem_q_w(h)
        sem_k = self.sem_k_w(h)
        sem_v = self.sem_v_w(h)
        sem_attn = (sem_q @ sem_k.transpose(-2, -1)).softmax(dim=-1)
        Adj_sem = self.coefficient*sem_attn +(1-self.coefficient)*A_static
        Adj_sem.masked_fill_(self.sem_mask.to(z.device), float('-inf'))
        Adj_sem = Adj_sem.softmax(dim=-1)

        ht_geo = self.AGCRN( z, geo_v,self.node_embeddings, Adj_geo)
        ht_sem = self.AGCRN( z, sem_v,self.node_embeddings, Adj_sem)

       
        z = torch.cat([ht_geo, ht_sem ], dim=-1)
        z = self.mlp(z)

        return z
    
    def self_att_chebnet(self, z):
        #下动态图和静态图分别做chebnet
        B, N, D = z.shape
        supports_embedding = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        geo_q = self.geo_q_w(z)
        sem_q = self.sem_q_w(z)
        geo_k = self.geo_k_w(z)
        sem_k = self.sem_k_w(z)
        
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale
        sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
        ##mask
        if self.geo_mask is not None:
            geo_attn.masked_fill_(self.geo_mask.to(z.device), float('-inf'))
        geo_attn = geo_attn.softmax(dim=-1)
        #64 170 170
        if self.sem_mask is not None:
            sem_attn.masked_fill_(self.sem_mask.to(z.device), float('-inf'))
        sem_attn = sem_attn.softmax(dim=-1)
     
        z_static = self.chebnet(supports_embedding.repeat(B,1,1),z)
        z_geo = self.chebnet(geo_attn,z)
        z_sem = self.chebnet(sem_attn,z)
        z = torch.sum(torch.stack([z_geo, z_sem, z_static], dim=-1), dim=-1)
        z = self.proj(z)
        return z

    

    def chebnet(self,supports, z):
        
        laplacian=False
        node_num = z.shape[1]
        supports_eye = torch.eye(node_num).unsqueeze(0).repeat(supports.shape[0],1,1).to(supports.device)
        if laplacian == True:
            support_set = [supports, -supports_eye]
        else:
            support_set = [supports_eye, supports]
        #default cheb_k = 2
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])

        z_g = []
        for support in support_set:
            z_g.append(torch.matmul(support, z) )
        #list 2[64 307 128 ]
        z_g = torch.stack(z_g, dim=0)
        #2 64 307 128 
        # supports = torch.stack(support_set, dim=0)
        # # 2 64 307 307
        z_g = z_g.permute(1,2,0,3)
        # # B, N, cheb_k, dim_in 64 307 2 128
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  
        #N, cheb_k, dim_in, dim_out 307 2 128 128
        bias = torch.matmul(self.node_embeddings, self.bias_pool)                       
        #N, dim_out 307 128
        z_g = torch.einsum('bnki,nkio->bno', z_g, weights) + bias
        return z_g

    def former(self, z,h):
        #z 64 170 64 B N I
        B, N, D = z.shape
    
        supports_embedding = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        z_static = self.chebnet(supports_embedding.repeat(B,1,1),z)

        #geo
        geo_q = self.geo_q_w(h)
        geo_k = self.geo_k_w(h)
        geo_v = self.geo_v_w(h)
        # geo_q = torch.einsum('bni,io->bno', h, self.geo_q_w) 
        # geo_k = torch.einsum('bni,io->bno', h, self.geo_k_w)
        # geo_v = torch.einsum('bni,io->bno', h, self.geo_v_w)
        geo_q = geo_q.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        geo_k = geo_k.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        geo_v = geo_v.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale

        if self.geo_mask is not None:
            geo_attn.masked_fill_(self.geo_mask.to(z.device), float('-inf'))
        geo_attn = geo_attn.softmax(dim=-1)
        #geo_attn = self.geo_att_drop(geo_attn)
        z_geo = (geo_attn @ geo_v).transpose(1, 2).reshape(B, N, int(D * self.geo_ratio))
        #sem
        sem_q = self.sem_q_w(h)
        sem_k = self.sem_k_w(h)
        sem_v = self.sem_v_w(h)
        # sem_q = torch.einsum('bni,io->bno', h, self.sem_q_w) 
        # sem_k = torch.einsum('bni,io->bno', h, self.sem_k_w)
        # sem_v = torch.einsum('bni,io->bno', h, self.sem_v_w)
        sem_q = sem_q.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        sem_k = sem_k.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        sem_v = sem_v.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
        
        if self.sem_mask is not None:
            sem_attn.masked_fill_(self.sem_mask.to(z.device), float('-inf'))
        sem_attn = sem_attn.softmax(dim=-1)
        #sem_attn = self.sem_att_drop(sem_attn)
        z_sem = (sem_attn @ sem_v).transpose(1, 2).reshape(B, N, int(D * self.geo_ratio))
        
        z_dynamic = self.proj(torch.cat([z_geo, z_sem], dim=-1))
        z_dynamic = self.mlp(z_dynamic)
        #z_dynamic = self.porj_drop(z_dynamic)
        z = torch.sum(torch.stack([z_dynamic, z_static], dim=-1), dim=-1)
        return z
    def former_GRU(self, z,h):
        #z 64 170 64 B N I
        B, N, D = z.shape
    
        supports_embedding = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        z_static = self.AGCRN(h, z, self.node_embeddings, supports_embedding)

        #geo
        geo_q = self.geo_q_w(h)
        geo_k = self.geo_k_w(h)
        geo_v = self.geo_v_w(h)
        geo_q = geo_q.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        geo_k = geo_k.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        geo_v = geo_v.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale

        if self.geo_mask is not None:
            geo_attn.masked_fill_(self.geo_mask.to(z.device), float('-inf'))
        geo_attn = geo_attn.softmax(dim=-1)
        #geo_attn = self.geo_att_drop(geo_attn)
        z_geo = (geo_attn @ geo_v).transpose(1, 2).reshape(B, N, int(D * self.geo_ratio))
        #sem
        sem_q = self.sem_q_w(h)
        sem_k = self.sem_k_w(h)
        sem_v = self.sem_v_w(h)
        sem_q = sem_q.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        sem_k = sem_k.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        sem_v = sem_v.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
        
        if self.sem_mask is not None:
            sem_attn.masked_fill_(self.sem_mask.to(z.device), float('-inf'))
        sem_attn = sem_attn.softmax(dim=-1)
        #sem_attn = self.sem_att_drop(sem_attn)
        z_sem = (sem_attn @ sem_v).transpose(1, 2).reshape(B, N, int(D * self.geo_ratio))
        
        z_dynamic = self.proj(torch.cat([z_geo, z_sem], dim=-1))
        z_dynamic = self.mlp(z_dynamic)
        #z_dynamic = self.porj_drop(z_dynamic)
        z = torch.sum(torch.stack([z_dynamic, z_static], dim=-1), dim=-1)
        return z

class STEncoderBlock(nn.Module):
    def __init__(self,hidden_channels=64, geo_num_heads=4, sem_num_heads=4,nums_node=170,embed_dim=10,cheb_k=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.mlp = Mlp(in_features=hidden_channels, hidden_features=hidden_channels, act_layer=nn.GELU, drop=0.)
        self.st_attn = STSelfAttention(geo_num_heads=geo_num_heads, 
                                       sem_num_heads=sem_num_heads, 
                                       hidden_channels=hidden_channels, 
                                       hidden_hidden_channels=hidden_channels, 
                                       num_nodes=nums_node, 
                                       embed_dim=embed_dim, 
                                       cheb_k=cheb_k)

    def forward(self, z, h):
        z =  self.st_attn(z, self.norm1(h))
        z = z + self.mlp(self.norm2(z))
        return z

class STSelfAttention(nn.Module):
    def __init__(self, geo_num_heads=4, sem_num_heads=4, hidden_channels=64,hidden_hidden_channels=64,
                  num_nodes=170, embed_dim=10, cheb_k=2):
        super().__init__()

        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.head_dim = hidden_channels // (self.geo_num_heads + self.sem_num_heads)
         #   hidden_channels 64 self.head_dim 8
        self.scale = self.head_dim ** -0.5
        self.geo_ratio = self.geo_num_heads / (self.geo_num_heads + self.sem_num_heads)
        self.sem_ratio = self.sem_num_heads / (self.geo_num_heads + self.sem_num_heads)
        self.enbeddings_ratio = 1 - self.geo_ratio - self.sem_ratio
        self.geo_attn_drop = nn.Dropout(0.1)
        self.sem_attn_drop = nn.Dropout(0.1)
        self.geo_q_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
        self.geo_k_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
        self.geo_v_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
        self.sem_q_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
        self.sem_k_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
        self.sem_v_w = torch.nn.Linear(hidden_channels, int(hidden_channels*self.geo_ratio))
        self.proj = nn.Linear(hidden_channels, hidden_channels)
        # self.norm1 = nn.LayerNorm(hidden_channels)
        # self.norm2 = nn.LayerNorm(hidden_channels)
        self.mlp = Mlp(in_features=hidden_channels, hidden_features=hidden_channels, act_layer=nn.GELU, drop=0.)


        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))
        self.geo_mask = torch.load("./data/PEMS08/geo_mask.pth")
        self.sem_mask = torch.load("./data/PEMS08/geo_mask.pth")
    def forward(self, z, h):
      
        #z 64 170 64 B N I
        B, N, D = z.shape
    
        supports_embedding = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        z_static = self.chebnet(supports_embedding.repeat(B,1,1),z)

        #geo
        geo_q = self.geo_q_w(h)
        geo_k = self.geo_k_w(h)
        geo_v = self.geo_v_w(h)
        geo_q = geo_q.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        geo_k = geo_k.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        geo_v = geo_v.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale

        if self.geo_mask is not None:
            geo_attn.masked_fill_(self.geo_mask.to(z.device), float('-inf'))
        geo_attn = geo_attn.softmax(dim=-1)
        #geo_attn = self.geo_att_drop(geo_attn)
        z_geo = (geo_attn @ geo_v).transpose(1, 2).reshape(B, N, int(D * self.geo_ratio))
        #sem
        sem_q = self.sem_q_w(h)
        sem_k = self.sem_k_w(h)
        sem_v = self.sem_v_w(h)
        sem_q = sem_q.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        sem_k = sem_k.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        sem_v = sem_v.reshape(B,  N, self.geo_num_heads, self.head_dim).permute(0, 2, 1, 3)
        sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
        
        if self.sem_mask is not None:
            sem_attn.masked_fill_(self.sem_mask.to(z.device), float('-inf'))
        sem_attn = sem_attn.softmax(dim=-1)
        #sem_attn = self.sem_att_drop(sem_attn)
        z_sem = (sem_attn @ sem_v).transpose(1, 2).reshape(B, N, int(D * self.geo_ratio))
        
        z_dynamic = self.proj(torch.cat([z_geo, z_sem], dim=-1))
        z_dynamic = self.mlp(z_dynamic)
        #z_dynamic = self.porj_drop(z_dynamic)
        z = torch.sum(torch.stack([z_dynamic, z_static], dim=-1), dim=-1)
        return z
        
    def chebnet(self,supports, z):
        laplacian=False
        node_num = z.shape[1]
        supports_eye = torch.eye(node_num).unsqueeze(0).repeat(supports.shape[0],1,1).to(supports.device)
        if laplacian == True:
            support_set = [supports, -supports_eye]
        else:
            support_set = [supports_eye, supports]
        #default cheb_k = 2
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])

        z_g = []
        for support in support_set:
            z_g.append(torch.matmul(support, z) )
        #list 2[64 307 128 ]
        z_g = torch.stack(z_g, dim=0)
        #2 64 307 128 
        # supports = torch.stack(support_set, dim=0)
        # # 2 64 307 307
        z_g = z_g.permute(1,2,0,3)
        # # B, N, cheb_k, dim_in 64 307 2 128
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  
        #N, cheb_k, dim_in, dim_out 307 2 128 128
        bias = torch.matmul(self.node_embeddings, self.bias_pool)                       
        #N, dim_out 307 128
        z_g = torch.einsum('bnki,nkio->bno', z_g, weights) + bias
        return z_g
    


    


class VectorField_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type):
        super(VectorField_g, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        
        # self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        #                                    for _ in range(num_hidden_layers - 1))

        #FIXME:
        
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels)  
        
        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))


    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        if self.g_type == 'agc':
            z = self.agc(z)
        else:
            raise ValueError('Check g_type argument')
        # for linear in self.linears:
        #     z = linear(x_gconv)
        #     z = z.relu()

        #FIXME:
        # z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
        z = z.tanh()
        return z #torch.Size([64, 307, 128, 128])


    def agc(self, z):
        """
        Adaptive Graph Convolution
        - Node Adaptive Parameter Learning
        - Data Adaptive Graph Generation
        """




        node_num = self.node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
            # laplacian=False
        laplacian=False
        if laplacian == True:
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            support_set = [supports, -torch.eye(node_num).to(supports.device)]
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            # support_set = [-supports]
        else:
            support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
         # supports：2  307 307
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  
        #N, cheb_k, dim_in, dim_out 307 2 128 128
        bias = torch.matmul(self.node_embeddings, self.bias_pool)                       
        #N, dim_out 307 128
        x_g = torch.einsum("knm,bmc->bknc", supports, z)      
        #B, cheb_k, N, dim_in 64 2 307 128
        x_g = x_g.permute(0, 2, 1, 3)  
        # B, N, cheb_k, dim_in 64 307 2 128
        z = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     
        #b, N, dim_out
        return z


class VectorField_only_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type):
        super(VectorField_only_g, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        
        # self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        #                                    for _ in range(num_hidden_layers - 1))

        #FIXME:
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        # self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        
        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))


    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        if self.g_type == 'agc':
            z = self.agc(z)
        else:
            raise ValueError('Check g_type argument')
        # for linear in self.linears:
        #     z = linear(x_gconv)
        #     z = z.relu()

        #FIXME:
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        # z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
        z = z.tanh()
        return z #torch.Size([64, 307, 64, 1])

    def agc(self, z):
        """
        Adaptive Graph Convolution
        - Node Adaptive Parameter Learning
        - Data Adaptive Graph Generation
        """
        node_num = self.node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)

        laplacian=False
        if laplacian == True:
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            support_set = [supports, -torch.eye(node_num).to(supports.device)]
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            # support_set = [-supports]
        else:
            support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, z)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        z = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return z

class VectorField_g_prime(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type):
        super(VectorField_g_prime, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        
        # self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        #                                    for _ in range(num_hidden_layers - 1))

        self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        
        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))


    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        if self.g_type == 'agc':
            z = self.agc(z)
        else:
            raise ValueError('Check g_type argument')
        # for linear in self.linears:
        #     z = linear(x_gconv)
        #     z = z.relu()

        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        z = z.tanh()
        return z #torch.Size([64, 307, 64, 1])

    def agc(self, z):
        """
        Adaptive Graph Convolution
        - Node Adaptive Parameter Learning
        - Data Adaptive Graph Generation
        """
        node_num = self.node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, z)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        z = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return z