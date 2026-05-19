import os
import ray
import yaml
import torch
import argparse
from tqdm import tqdm

from evolutionary_rlhf.population import Population
from evolutionary_rlhf.individual import Individual
from evolutionary_rlhf.diffusion_reward import DiffusionRewardModel
from evolutionary_rlhf.data_utils import load_dataset

class EvolutionTrainer:
    def __init__(self, configs):
        self.evo_cfg = configs['evolution']
        self.ppo_cfg = configs['ppo']
        self.model_cfg = configs['models']
        self.diff_cfg = configs['diffusion']
        self.task_cfg = configs['task']

        self.population_size = self.evo_cfg['population_size']
        self.num_generations = self.evo_cfg['num_generations']
        self.elite_ratio = self.evo_cfg['elite_ratio']
        self.top_k = self.evo_cfg['top_k_for_diffusion']

        self.dataset = load_dataset(self.task_cfg)
        self.diffusion = None

    def form_initial_population(self):
        print("Forming initial population...")
        population = Population()
        for i in range(self.population_size):
            ind = Individual(id=i, model_cfg=self.model_cfg, ppo_cfg=self.ppo_cfg)
            trajectories, tau = ind.generate_with_hidden_states(self.dataset, self.task_cfg)
            if self.diffusion is None:
                self.diffusion = DiffusionRewardModel(self.diff_cfg)
            rewards = self.diffusion.conditionally_generate(tau)
            ind.set_training_data(trajectories, rewards)
            ind.ppo_update()
            rho = ind.evaluate_orm(self.dataset, self.task_cfg)
            ind.rho = rho
            population.add(ind)
        return population

    def evolution_loop(self, population):
        for gen in range(self.num_generations):
            print(f"\n=== Generation {gen+1}/{self.num_generations} ===")
            # 5.1 train diffusion on elites
            elite_tau, elite_R = population.get_top_k_tau_R(self.top_k)
            self.diffusion.train(elite_tau, elite_R)

            # 5.2 select elites
            elites = population.select_elite(self.elite_ratio)
            new_individuals = []

            for elite in tqdm(elites, desc="Mutating elites"):
                # 5.3.1 truncated diffusion -> mutated reward R'
                R_prime = self.diffusion.truncated_diffusion(
                    elite.tau, elite.R,
                    steps=self.evo_cfg['mutation']['truncation_steps'],
                    noise=self.evo_cfg['mutation']['noise_level']
                )
                # 5.3.2-5.3.3 train temp model from elite with R'
                temp_ind = elite.clone()
                temp_ind.set_training_data(elite.trajectories, R_prime)
                temp_ind.ppo_update()
                # 5.3.4 generate new trajectories
                new_traj, tau_new = temp_ind.generate_with_hidden_states(self.dataset, self.task_cfg)
                # 5.3.5 alignment: diffusion scores new tau -> R_new
                R_new = self.diffusion.conditionally_generate(tau_new)
                # 5.3.6-5.3.7 train again with clean reward
                temp_ind.set_training_data(new_traj, R_new)
                temp_ind.ppo_update(epochs=self.evo_cfg['mutation']['alignment_ppo_epochs'])
                # 5.3.8 evaluate new ORM score
                new_rho = temp_ind.evaluate_orm(self.dataset, self.task_cfg)
                temp_ind.rho = new_rho
                temp_ind.tau = tau_new
                temp_ind.R = R_new
                new_individuals.append(temp_ind)

            # 5.4 add to population
            population.add_all(new_individuals)
            # 5.5 keep top M
            population.prune(self.population_size)
            # 5.6 record best
            best_rho = population.best_rho()
            print(f"Generation {gen+1} best ρ: {best_rho:.4f}")

        return population

    def run(self):
        ray.init()
        population = self.form_initial_population()
        final_pop = self.evolution_loop(population)
        print("Evolution completed. Best individual ρ =", final_pop.best_rho())
        ray.shutdown()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evolution_config", type=str, required=True)
    parser.add_argument("--ppo_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--diffusion_config", type=str, required=True)
    parser.add_argument("--task_config", type=str, required=True)
    parser.add_argument("--init_only", action="store_true")
    args = parser.parse_args()

    configs = {
        'evolution': yaml.safe_load(open(args.evolution_config)),
        'ppo': yaml.safe_load(open(args.ppo_config)),
        'models': yaml.safe_load(open(args.model_config)),
        'diffusion': yaml.safe_load(open(args.diffusion_config)),
        'task': yaml.safe_load(open(args.task_config)),
    }
    trainer = EvolutionTrainer(configs)
    trainer.run()

if __name__ == "__main__":
    main()
