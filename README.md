# Analysis-of-motion-patterns-with-adhoc-networks
summary
This code can be used to run a simulation of an ad hoc network. Ad hoc networks are wireless networks composed of a set of interconnected nodes without a specific routing infrastructure. In this code, it is assumed that the nodes move according to a specific movement model and have a certain transmission range. Additionally, by simulating message transmission between nodes at a specific time, it calculates and visualizes the rate of messages transmitted in the network.
The code includes two classes, AdHocNode and AdHocNetwork. The AdHocNode class is represented by a node's unique identifier (node_id) and coordinates (x and y). The AdHocNetwork class comes with a node list consisting of AdHocNode instances. It also has a specific movement model (move_model_type) and a specific transmission range (transmission_range). When a simulation is run, nodes are randomly moved using the movement model and then message exchange between nodes is simulated.
The code also includes an abstract class called MovementModel to represent different movement models. Subclasses derived from this abstract class implement different movement models. For example, the RandomMovement class randomly moves nodes within a specified maximum number of steps, while the GaussMarkov class determines the current movement based on a measure of previous movement.
The code also uses the matplotlib library to visualize simulation results.