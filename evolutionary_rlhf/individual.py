import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from evolutionary_rlhf.data_utils import compute_orm

class Individual:
    def __init__(self, id, model_cfg, ppo_cfg):
        self.id = id
        self.model_cfg = model_cfg
        self.ppo_cfg = ppo_cfg
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg['base_policy'])
        self.trajectories = None
        self.tau = None   # hidden states
        self.R = None     # reward sequence
        self.rho = 0.0

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_cfg['base_policy'])
        if self.model_cfg.get('use_lora', False):
            lora_config = LoraConfig(
                r=self.model_cfg['lora_rank'],
                lora_alpha=self.model_cfg['lora_alpha'],
                target_modules=["q_proj", "v_proj"],
            )
            model = get_peft_model(model, lora_config)
        return model

    def generate_with_hidden_states(self, dataset, task_cfg):
        # TODO: integrate with OpenRLHF's generation (vLLM) and hook hidden states
        seq_len = task_cfg.get('max_new_tokens', 256)
        hidden_dim = self.model_cfg.get('hidden_dim', 3584)
        tau = torch.randn(len(dataset), seq_len, hidden_dim)   # placeholder
        trajectories = [{"context": d['question'], "response": "dummy answer"} for d in dataset]
        return trajectories, tau

    def set_training_data(self, trajectories, rewards):
        self.trajectories = trajectories
        self.R = rewards

    def ppo_update(self, epochs=None):
        # TODO: call OpenRLHF's PPO trainer with custom reward (using self.R)
        pass

    def evaluate_orm(self, dataset, task_cfg):
        return compute_orm(self.model, self.tokenizer, dataset, task_cfg)

    def clone(self):
        new_ind = Individual(id=self.id+1000, model_cfg=self.model_cfg, ppo_cfg=self.ppo_cfg)
        new_ind.model.load_state_dict(self.model.state_dict())
        new_ind.trajectories = self.trajectories
        new_ind.tau = self.tau
        new_ind.R = self.R
        new_ind.rho = self.rho
        return new_ind
