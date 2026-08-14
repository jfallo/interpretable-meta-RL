import torch


class BanditsEnv:
    def __init__(self, config, device):
        self.D = config['D']
        self.num_arms = config['num_arms']
        self.dependent_arms = config['dependent_arms']
        self.restless = config['restless']
        self.drift = config['drift']
        self.T = config['num_trials']

        self.device = device

        self.set_batch_size()
        self.reset(active_models= {})


    def set_batch_size(self, batch_size= 1):
        self.batch_size = batch_size
        self.batch_idx = torch.arange(self.batch_size, device= self.device)


    def build_models(self, models):
        self.models = models


    def reset(self, active_models, testing= False):
        # reset task
        self.probs = self.D(self.batch_size, self.num_arms, device= self.device)

        # reset states
        self.h, self.c, self.x = {}, {}, {}
        if 'DisRNN' in active_models:
            self.h['DisRNN'] = torch.zeros(self.batch_size, self.models['DisRNN']['hidden_size'], device= self.device)
            self.x['DisRNN'] = torch.zeros(self.batch_size, self.models['DisRNN']['input_size'], device= self.device)
        if 'DisLRU' in active_models:
            self.h['DisLRU'] = torch.zeros(self.batch_size, self.models['DisLRU']['hidden_size'], device= self.device)
            self.x['DisLRU'] = torch.zeros(self.batch_size, self.models['DisLRU']['input_size'], device= self.device)
        if 'LSTM' in active_models:
            self.h['LSTM'] = torch.zeros(1, self.batch_size, self.models['LSTM']['hidden_size'], device= self.device)
            self.c['LSTM'] = torch.zeros(1, self.batch_size, self.models['LSTM']['hidden_size'], device= self.device)
            self.x['LSTM'] = torch.zeros(self.batch_size, self.models['LSTM']['input_size'], device= self.device)

        if testing:
            self.models['Thompson']['model'].reset()
            self.models['UCB']['model'].reset()
            self.models['Gittins']['model'].reset()


    def step(self, active_models):
        logits = {model: None for model in active_models}
        kls = {
            model: {bottleneck: None for bottleneck in self.models[model]['bottlenecks']} 
            for model in active_models if 'bottlenecks' in self.models[model]
        }
        critic_inputs = {model: None for model in active_models}
        if 'DisRNN' in active_models:
            self.h['DisRNN'], kls['DisRNN'] = self.models['DisRNN']['model'].step(self.h['DisRNN'], self.x['DisRNN'])
            logits['DisRNN'] = self.models['DisRNN']['model'].out(self.h['DisRNN'])
            critic_inputs['DisRNN'] = self.h['DisRNN'].detach()
        if 'DisLRU' in active_models:
            self.h['DisLRU'], kls['DisLRU'] = self.models['DisLRU']['model'].step(self.h['DisLRU'], self.x['DisLRU'])
            logits['DisLRU'] = self.models['DisLRU']['model'].out(self.h['DisLRU'])
            critic_inputs['DisLRU'] = self.h['DisLRU'].detach()
        if 'LSTM' in active_models:
            out, (self.h['LSTM'], self.c['LSTM']) = self.models['LSTM']['model'](self.x['LSTM'].unsqueeze(0), (self.h['LSTM'], self.c['LSTM']))
            logits['LSTM'] = self.models['LSTM']['readout'](out.squeeze(0))
            critic_inputs['LSTM'] = out.squeeze(0)

        return logits, kls, critic_inputs


    def sample(self, active_models, logits, classical_models= {'Thompson', 'UCB', 'Gittins'}):
        arm_rewards = torch.bernoulli(self.probs)

        pi = {model: None for model in active_models}
        a = {model: None for model in active_models}
        r = {model: None for model in active_models}
        for model in active_models:
            if model in classical_models:
                a[model] = self.models[model]['model'].choice()
                r[model] = arm_rewards[0, a[model]].item()
                self.models[model]['model'].getReward(a[model], r[model])
            else:
                pi[model] = torch.distributions.Categorical(logits= logits[model])
                a[model] = pi[model].sample()
                r[model] = arm_rewards[self.batch_idx, a[model]]

        return pi, a, r


    def update(self, active_models, a, r):
        # restless bandits
        if self.restless:
            self.probs += self.drift * torch.randn(self.batch_size, self.num_arms, device= self.device)
            self.probs = torch.clamp(self.probs, 0, 1)
            if self.dependent_arms:
                self.probs[:, 1] = 1 - self.probs[:, 0]

        # obs
        if 'DisRNN' in active_models:
            self.x['DisRNN'] = torch.stack([2*a['DisRNN'].float() - 1, 2*r['DisRNN'] - 1], dim= -1)
        if 'DisLRU' in active_models:
            self.x['DisLRU'] = torch.zeros(self.batch_size, self.models['DisLRU']['input_size'], device= self.device)
            self.x['DisLRU'][torch.arange(self.batch_size, device= self.device), a['DisLRU']] = 2*r['DisLRU'] - 1
        if 'LSTM' in active_models:
            self.x['LSTM'] = torch.stack([2*a['LSTM'].float() - 1, 2*r['LSTM'] - 1], dim= -1)


    def training_episode(self, active_models, steps_unrolled):
        log_probs = {model: [] for model in active_models}
        rewards = {model: [] for model in active_models}
        expected_returns = {model: [] for model in active_models}
        entropies = {model: [] for model in active_models}
        regrets = {model: [] for model in active_models}
        bottleneck_losses = {
            model: {bottleneck: [] for bottleneck in self.models[model]['bottlenecks']} 
            for model in active_models if 'bottlenecks' in self.models[model]
        }

        for t in range(self.T):
            # detach gradients
            if t % steps_unrolled == 0:
                for model in self.h:
                    self.h[model] = self.h[model].detach()
                for model in self.c:
                    self.c[model] = self.c[model].detach()

            # step
            logits, kls, critic_inputs = self.step(active_models)
            # sample
            pi, a, r = self.sample(active_models, logits)

            optimal = self.probs.max(dim= -1).values
            for model in active_models:
                log_probs[model].append(pi[model].log_prob(a[model]))
                rewards[model].append(r[model])
                entropies[model].append(pi[model].entropy())
                expected_returns[model].append(self.models[model]['critic'](critic_inputs[model]).squeeze(-1))
                regrets[model].append(optimal - self.probs[self.batch_idx, a[model]])
                if model in bottleneck_losses:
                    for key, val in kls[model].items():
                        bottleneck_losses[model][key].append(val)

            # update
            self.update(active_models, a, r)

        return (
            {model: torch.stack(vals) for model, vals in rewards.items()},
            {model: torch.stack(vals) for model, vals in expected_returns.items()},
            {model: torch.stack(vals) for model, vals in log_probs.items()},
            {model: torch.stack(vals) for model, vals in entropies.items()},
            {model: torch.stack(vals) for model, vals in regrets.items()},
            {
                model: {key: torch.stack(vals) for key, vals in bottleneck_losses[model].items()}
                for model in bottleneck_losses
            }
        )


    def eval_episode(self, active_models):
        regrets = {model: [] for model in active_models}

        for _ in range(self.T):
            # step
            logits, _, _ = self.step(active_models)
            # sample
            _, a, r = self.sample(active_models, logits)

            optimal = self.probs.max(dim= -1).values
            for model in active_models:
                regrets[model].append(optimal - self.probs[self.batch_idx, a[model]])

            # update
            self.update(active_models, a, r)

        return {model: torch.stack(vals) for model, vals in regrets.items()}
