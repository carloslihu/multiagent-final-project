from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules.NetworkVisualization import NetworkModule
from mesa.visualization.modules import TextElement

from model import TSPModel


# def network_portrayal(G):
#     # The model ensures there is always 1 agent per node

#     def node_color():
#         return "#008000" # green

#     def edge_color():
#         return "#000000" # black

#     def edge_width(source, target):
#         e = (source, target)
#         if G.get_edge_data(*e)['pheromone'] > 1:
#             return 3
#         else:
#             return 1

#     portrayal = dict()
#     portrayal["nodes"] = [
#         {
#             "id": node_id,
#             "size": 3 if agents else 1,
#             "color": "#CC0000" if not agents else "#007959",
#             "label": "Node {}".format(node_id),
#             "x": G.nodes[node_id]['coord'][0],
#             "y": G.nodes[node_id]['coord'][1]
#         }
#         for (node_id, agents) in G.nodes.data("agent")
#     ]

#     portrayal["edges"] = [
#         {"id": edge_id, "source": source, "target": target, "color": "#000000"}
#         for edge_id, (source, target) in enumerate(G.edges)
#     ]

#     return portrayal


# network = NetworkModule(network_portrayal, 500, 500, library="sigma")


chart = ChartModule(
    [
        {"Label": "L_best", "Color": "#FF0000"},
    ], data_collector_name="datacollector"
)

class MyTextElement(TextElement):
    def render(self, model):
        global_best_length = model.global_best_L
        global_best_path = model.global_best_path
        return "Global Best Length: {}<br>Global best path: {}".format(
            global_best_length, global_best_path
        )

model_params = {
    "a": UserSettableParameter(
        "slider",
        "History Coefficient",
        1,
        0,2,
        0.1,
        description="Choose History Coefficient in the model",
    ),
    "b": UserSettableParameter(
        "slider",
        "Heuristic Coefficient",
        2,
        0,6,
        1,
        description="Choose Heuristic Coefficient in the model",
    ),
    "ro": UserSettableParameter(
        "slider",
        "Evaporation Rate",
        0.02,
        0,0.5,
        0.01,
        description="Choose Evaporation Rate in the model",
    ),
    "m": UserSettableParameter(
        "slider",
        "Number of agents",
        30,
        1,100,
        1,
        description="Choose how many agents to include in the model",
    ), 
    "tao_init": UserSettableParameter(
        "slider",
        "Initial Pheromone",
        5,
        0,10,
        0.5,
        description="Choose Initial Pheromone to include in the model",
    ), 
    "tao_max": UserSettableParameter(
        "slider",
        "Pheromone Upper Bound",
        25,
        2,50,
        1,
        description="Choose Pheromone Upper Bound to include in the model",
    ),
    "tao_min": UserSettableParameter(
        "slider",
        "Pheromone Lower Bound",
        2.5,
        0,5,
        0.5,
        description="Choose Pheromone Lower Bound to include in the model",
    ), 
}

server = ModularServer(
    TSPModel, [
#         network, 
        MyTextElement(),
        chart
    ], "TSP Model", model_params
)
server.port = 8521