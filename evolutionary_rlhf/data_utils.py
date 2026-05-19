from datasets import load_dataset

def load_dataset(task_cfg):
    dataset = load_dataset(task_cfg['dataset'], split='train')
    dataset = dataset.select(range(task_cfg['num_prompts_per_individual']))
    return dataset

def extract_hidden_states(model, tokenizer, prompts):
    # TODO: implement forward hooks to collect hidden states
    return None

def compute_orm(model, tokenizer, dataset, task_cfg):
    # TODO: implement exact match or math equivalence evaluation
    return 0.0
