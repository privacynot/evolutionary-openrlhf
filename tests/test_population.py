def test_population_add():
    from evolutionary_rlhf.population import Population
    pop = Population()
    assert len(pop.individuals) == 0
