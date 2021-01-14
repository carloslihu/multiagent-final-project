import numpy as np
import pandas as pd
import tsplib95
import itertools

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector


################################# GENERAL FUNCTIONS ###########################################
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def get_best_length(model):
    """Picks the shortest length from the model (in this iteration)"""
    agent_lengths = [agent.total_distance for agent in model.schedule.agents]
    L_best = min(agent_lengths)
    return L_best


def get_best_agents(model):
    """Picks the agent with the shortest length from the model (in this iteration)"""
    L_best = get_best_length(model)
    
    agents = model.schedule.agents
    list_agents = [[a.unique_id, a.visited_nodes, a.total_distance] for a in agents if a.total_distance == L_best]
    
    best_agents = pd.DataFrame(list_agents, columns = ['unique_id', 'visited_nodes', 'total_distance'])
    return best_agents
############################################################################


#################################### MODEL ########################################
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
        self.G = self.grid.G
        self.max_nodes = self.grid.G.number_of_nodes()
        
        # Initialize pheromones to each edge (edges are bidirectional and share attributes)
        for (u, v) in self.grid.G.edges:
            e = (u,v)
            self.grid.G.get_edge_data(*e)['pheromone'] = self.pheromone_initial
        
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={"L_best": get_best_length
                             # ,"best_agent": get_best_agents
                            }, # add best path and best agent?
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
        
        # TODO comment
        # self.datacollector.collect(self)

    def pheromone_update(self, tao, L_best):
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
        
        ro = self.evaporation_rate
        tao_max = self.pheromone_max
        tao_min = self.pheromone_min
        
        # formula code
        tao_best = pheromone_quantity(L_best)
        pheromone = max_min(
            (1 - ro) * tao + tao_best,
            tao_max, tao_min
        )
        return pheromone
    
    def reset_agents(self):
        for a in self.schedule.agents:
            # restart attributes
            a.visited_nodes = []
            a.total_distance = 0
            # select initial random grid for next iteration
            random_node_id = self.random.randint(1, self.max_nodes)
            self.grid.place_agent(a, random_node_id)
            a.visited_nodes.append(random_node_id)
            
    def step(self):
        # Each step is a complete solution
        
        # TODO 1 complete cycle
        for i in range(self.max_nodes):
            self.schedule.step()
        
        # collect data
        self.datacollector.collect(self)
        
        # obtain L_best from this iteration
        # L_best = self.datacollector.get_model_vars_dataframe().tail(1)['L_best'].iloc[0]        
        # best_agent = self.datacollector.get_model_vars_dataframe().tail(1)['best_agent'].iloc[0]
        
        # We obtain agents with the lowest total_distance from this iteration
        best_agents = get_best_agents(self)
        
        # We iterate for each best ant its pheromone update
        for index, row in best_agents.iterrows():
            L_best = row['total_distance']
            best_tour = row['visited_nodes']
            
            # We iterate for each best tour edge
            for u, v in pairwise(best_tour):
                e = (u, v)
                tao = self.grid.G.get_edge_data(*e)['pheromone']
                pheromone = self.pheromone_update(tao, L_best)
                self.grid.G.get_edge_data(*e)['pheromone'] = pheromone
        
        # Agents are reset for the next step
        self.reset_agents()
        
        
    def run_model(self, n):
        """Runs the model for n iterations"""
        for i in range(n):
            self.step()
############################################################################            



############################### AGENT #############################################
class TSPAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.visited_nodes = [] # node_id of visited nodes is stored
        self.total_distance = 0 # total travel distance

    def calculate_probabilities(self):
        # Calculates probabilities with formula
        
        # Obtains feasible nodes
        current_node = self.visited_nodes[-1]
        all_nodes = range(1, self.model.max_nodes + 1) # last element not included
        feasible_nodes = [x for x in all_nodes if x not in self.visited_nodes]
        
        # Initializes probabilities array
        probabilities = np.zeros(self.model.max_nodes)
        for f in feasible_nodes:
            
            e = (current_node, f) # feasible edge
            
            tao_ij = self.model.grid.G.get_edge_data(*e)['pheromone']
            a = self.model.history_coefficient
            
            eta =  1 / self.model.grid.G.get_edge_data(*e)['weight']
            b = self.model.heuristic_coefficient
            
            # Probability formula
            probabilities[f-1] = (tao_ij ** a) * (eta ** b)# f-1 because array starts at 0
        
        # divides by total sum
        if np.sum(probabilities) != 0:
            probabilities = probabilities / np.sum(probabilities)
        return probabilities
    
    def visit_node(self, probabilities):
        # Chooses unvisited node stochastically
        all_nodes = range(1, self.model.max_nodes + 1)
        if all(p == 0 for p in probabilities):
            next_node_id = all_nodes[0]
        else:
            next_node_id = self.model.random.choices(all_nodes, weights=probabilities, k=1)[0]
        
        # Updates the agent
        e = (self.visited_nodes[-1], next_node_id)
        self.total_distance = self.total_distance + self.model.grid.G.get_edge_data(*e)['weight']
        self.visited_nodes.append(next_node_id)
        
        # Updates the grid
        self.model.grid.move_agent(self, next_node_id)
        
    def step(self):
        probabilities = self.calculate_probabilities()
        self.visit_node(probabilities)