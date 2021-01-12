from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid

# from mesa.visualization.modules import NetworkVisualization
from mesa.datacollection import DataCollector

import tsplib95

def best_length(model):
    """Picks the shortest length from the model (in this iteration)"""
    agent_lengths = [agent.total_distance for agent in model.schedule.agents]
    L_best = min(agent_lengths)
    return L_best


class TSPModel(Model):
    def __init__(self, a=1, b=2, ro=0.02, m=10, tao_init=1, tao_max=2, tao_min=0#, width=10, height=10
                 , tsp_data_file = 'data/wi29.tsp'
                ):
        # Model parameters
        self.history_coefficient = a
        self.heuristic_coefficient = b
        self.evaporation_rate = ro
        self.num_agents = m
        self.pheromone_initial = tao_init
        self.pheromone_max = tao_max
        self.pheromone_min = tao_min
        
        # Reads the tsp_data_file
        problem = tsplib95.load(tsp_data_file)
        
        # the grid is created
        self.grid = NetworkGrid(problem.get_graph())
        self.max_nodes = self.grid.G.number_of_nodes()
        
        # Initialize pheromones to each edge
        for (u, v) in self.grid.G.edges:
            self.grid.G.edges[u,v]['pheromone'] = self.pheromone_initial
        # TODO treat as non-directed?
        
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={"L_best": best_length}, 
            agent_reporters={"Length": "total_distance"}
        )
        
        # Create agents and place them randomly in the grid
        for i in range(self.num_agents):
            a = TSPAgent(i, self)
            self.schedule.add(a)
            
            # Add each agent to a random grid cell and update visited_nodes
            random_node_id = self.random.randint(1, self.max_nodes)
            self.grid.place_agent(a, random_node_id) 
            a.visited_nodes.append(random_node_id)
        
        # batch parameter
        self.running = True
        
        # unnecessary?
        self.datacollector.collect(self)

    def pheromone_update(ro, tao, L_best, tao_max, tao_min):
        """Updates the pheromones in the graph each step"""
        def max_min(x, a, b):
            """Returns the max_min operator of x (e.g. [x]^a_b)"""
            if x > a:
                return a
            elif x < b:
                return b
            else:
                return x

        def pheromone_quantity(L_best):
            """Calculates the new best pheromone of this step"""
            tao_best = 1 / L_best
            return tao_best

        
        tao_best = pheromone_quantity(L_best)
        pheromone = max_min(
            (1 - ro) * tao + tao_best,
            tao_max, tao_min
        )
        return pheromone

    def step(self):
        # Each step is a complete solution
        for i in range(self.max_nodes):
            self.schedule.step()
        # collect data
        self.datacollector.collect(self)
        
        # TODO pheromone_update
        # TODO reset agents
        
    def run_model(self, n):
        """Runs the model for n iterations"""
        for i in range(n):
            self.step()

    
class TSPAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.visited_nodes = [] # node_id of visited nodes is stored
        self.total_distance = 0 # total travel distance

    def calculate_probabilities():
        # TODO calculate probabilities
        # self.model.history_coefficient
        return
    
    def visit_node():
        # TODO choose unvisited node stochastically
        
        # self.model.random.choices(numberList, weights=(10, 20, 30, 40, 50), k=1)
        # self.model.grid
        return


    def update_statistics():
        # update visited_nodes and total_distance
        return
    
    def step(self):
        
#         self.move()
#         if self.wealth > 0:
#             self.give_money()
        # TODO 
        # calculate_probabilities()
        # visit_node()
        # update_statistics
        return