import torch
import torch.nn as nn


def build_update_MLP(input_size, output_size= 2, hidden_size= 5):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )

def build_choice_MLP(input_size, output_size, hidden_size= 2):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )


class MyDisRNN(nn.Module):
    def __init__(self, m, n, q):
        super().__init__()

        self.m = m
        self.n = n
        self.q = q

        # build update MLPs and choice MLP
        self.updateMLPs = nn.ModuleList(
            [build_update_MLP(m+n) for _ in range(m)]
        )
        self.choiceMLP = build_choice_MLP(m,q)
            
        # initialize bottleneck parameters
        self.logit_M_h = nn.Parameter(2 * torch.ones((m,m)))  # M(i,j) -> update rule i's dependence on latent j
        self.log_sigma_h = nn.Parameter(-2 * torch.ones((m,m)))

        self.logit_M_x = nn.Parameter(2 * torch.ones((m,n)))
        self.log_sigma_x = nn.Parameter(-2 * torch.ones((m,n)))

        self.logit_M_z = nn.Parameter(2 *torch.ones(m))
        self.log_sigma_z = nn.Parameter(-2 * torch.ones(m))


    def bottleneck(self, x, m, sigma):
        if m.dim() == 1:
            noise = torch.randn_like(x) if self.training else torch.zeros_like(x)
            mean = m.unsqueeze(0) * x
            std = sigma.unsqueeze(0)
        else:
            shape = (m.shape[0], *x.shape)

            noise = torch.randn(shape, device= x.device) if self.training else torch.zeros(shape, device= x.device)
            mean = m.unsqueeze(1) * x.unsqueeze(0)
            std = sigma.unsqueeze(1)

        z = mean + std * noise
        kl = 0.5 * torch.sum(
            mean**2 + std**2 - torch.log(std**2 + 1e-8) - 1, 
            dim= -1
        )
        
        return z, kl

    def step(self, h, x):
        h_prev = h

        # bottlenecks for disentangled update rules
        M_h = torch.sigmoid(self.logit_M_h)
        sigma_h = torch.exp(self.log_sigma_h)
        h, kl_h = self.bottleneck(h, M_h, sigma_h)

        M_x = torch.sigmoid(self.logit_M_x)
        sigma_x = torch.exp(self.log_sigma_x)
        x, kl_x = self.bottleneck(x, M_x, sigma_x)

        # apply update MLPs
        z_outs = []
        for i, MLP in enumerate(self.updateMLPs):
            z_i = torch.cat([h[i], x[i]], dim= -1)
            logit_w, u = MLP(z_i).unbind(dim= -1)
            w = torch.sigmoid(logit_w)
            z_out = (1 - w) * h_prev[:, i] + w * u
            z_outs.append(z_out)
        z = torch.stack(z_outs, dim= -1)

        # bottlenecks for disentangled latents
        M_z = torch.sigmoid(self.logit_M_z)
        sigma_z = torch.exp(self.log_sigma_z)
        z, kl_z = self.bottleneck(z, M_z, sigma_z)

        kls = {
            'h': kl_h.sum(dim= 0),
            'x': kl_x.sum(dim= 0),
            'z': kl_z
        }

        return z, kls


    def out(self, h):
        return self.choiceMLP(h)


    def forward(self, X):
        T, B, n = X.shape
        assert n == self.n

        latents = []
        bottleneck_losses = {'h': [], 'x': [], 'z': []}
        outputs = []
        
        h = torch.zeros(B, self.m, device= X.device)
        latents.append(h)
        for t in range(T):
            x = X[t]

            h, kls = self.step(h, x)
            latents.append(h)
            for key, val in kls.items():
                bottleneck_losses[key].append(val)

            y = self.out(h)
            outputs.append(y)

        bottleneck_losses = {key: torch.stack(vals) for key, vals in bottleneck_losses.items()}

        return torch.stack(outputs), torch.stack(latents), bottleneck_losses
