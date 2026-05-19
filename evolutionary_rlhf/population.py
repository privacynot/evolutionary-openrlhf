import heapq

class Population:
    def __init__(self):
        self.individuals = []

    def add(self, ind):
        self.individuals.append(ind)

    def add_all(self, inds):
        self.individuals.extend(inds)

    def select_elite(self, elite_ratio):
        sorted_ind = sorted(self.individuals, key=lambda x: x.rho, reverse=True)
        k = max(1, int(len(self.individuals) * elite_ratio))
        return sorted_ind[:k]

    def get_top_k_tau_R(self, k):
        sorted_ind = sorted(self.individuals, key=lambda x: x.rho, reverse=True)[:k]
        taus = [ind.tau for ind in sorted_ind]
        Rs = [ind.R for ind in sorted_ind]
        return taus, Rs

    def prune(self, max_size):
        self.individuals = heapq.nlargest(max_size, self.individuals, key=lambda x: x.rho)

    def best_rho(self):
        return max(ind.rho for ind in self.individuals) if self.individuals else 0.0
