#DOCPLEX of the GC'17 ILP
from docplex.mp.model import Model
import simpy
import functools
import random as np
import time
from enum import Enum
from scipy.stats import norm
import matplotlib.pyplot as plt
#This ILP does the allocation of batches of RRHs to the processing nodes.
#It considers that each RRH is connected to the cloud and to only one fog node.

#log variables
power_consumption = []
execution_time = []
average_delay = []

#transmission delay to nodes
#considering a propagation time of 2*10^8 m/s and a core of mm 50µm - 20km to fog and 40km to cloud
fog_delay = 0.0000980654
cloud_delay = 0.0001961308

#number of fogs nodes
fog_amount = 5

#to keep the amount of sensors being processed on each node
sensors_on_nodes = [0,0,0,0,0,0]

#required transmission rate of each sensor
bandwidth_rate = 614.4
#required processing capacity of each sensor in processing units
sensor_units = 614.4;
#capacity of node in processing units
node_capacity = [98304, 19660.8, 19660.8, 19660.8, 19660.8, 19660.8]


#to assure that each lamba allocatedto a node can only be used on that node on the incremental execution of the ILP
lambda_node = []
'''
lambda_node = [
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
]
'''

#if a processing node (cloud or fog) is activated
nodeState = [0,0,0,0,0,0]

#general cost of each processing node
nodeCost = [
0.0,
300.0,
300.0,
300.0,
300.0,
300.0,
]

#cost of each transmission channel transceiver at a processing node
lc_cost = []

#capacity of each transmission channel
wavelength_capacity = []

#lc_cost = 20

#a very big number for modelling purposes
B = 1000000

lambda_state = []

#some input parameters example
#number of sensors
sensors = range(0,1)
#number of nodes
nodes = range(0, 6)
#number of lambdas
lambdas = range(0, 60)

#populate the wavelength data structures
def setLambdas():
	for i in lambdas:
		wavelength_capacity.append(10000.0)
		lambda_state.append(0)
		lc_cost.append(20.0)
		lambda_node.append([1,1,1,1,1,1])

#create the ilp formulation class
#this class is responsible for executing the ilp
class ILP(object):
	def __init__(self, sensor, sensors, nodes, lambdas):
		#input parameters
		self.sensor = sensor
		self.fog = []
		for i in sensor:
			self.fog.append(i.sensor_matrix)
		self.sensors = sensors
		self.nodes = nodes
		self.lambdas = lambdas
		#creates the ILP model/formulation itself
		self.setModel()

	#rexecutes the model
	def run(self):
		#self.setModel() #moved to the constructor to help when constraints need to be modified in execution time
		self.setConstraints()
		self.setObjective()
		sol = self.solveILP()
		return sol

	#creates the model and the decision variables
	def setModel(self):
		self.mdl = Model("Sensors transmission and processing scheduling")
		#indexes for the decision variables x[i,j,w]
		self.idx_ijw = [(i,j,w) for i in self.sensors for j in self.nodes for w in self.lambdas]
		self.idx_ij = [(i,j) for i in self.sensors for j in self.nodes]
		self.idx_wj = [(w, j) for w in self.lambdas for j in self.nodes]
		self.idx_j = [(j) for j in self.nodes]
		 
		#Decision variables
		self.x = self.mdl.binary_var_dict(self.idx_ijw, name = 'Sensor i is processed in node n with Lambda w', key_format = "")
		#y[rrhs][nodes];
		self.y = self.mdl.binary_var_dict(self.idx_ij, name = 'Sensors i is processed in node n')
		#xn[nodes]; Node n is activated
		self.xn = self.mdl.binary_var_dict(self.idx_j, name = 'Node n activated')
		#z[lambdas][nodes];Lambda w is allocated to node j
		self.z = self.mdl.binary_var_dict(self.idx_wj, name = 'Lambda w is transmited to node n')

	#create constraints
	def setConstraints(self):
		#each sensors can be processed only in one node and be transmited by only one transmission channel w
		self.mdl.add_constraints(self.mdl.sum(self.x[i,j,w] for j in self.nodes for w in self.lambdas) == 1 for i in self.sensors)#1
		#next constraints is used to avoid multiple sensors being transmited at the same transmission channel w - maybe does not make sense for our case at this moment
		#self.mdl.add_constraints(self.mdl.sum(self.x[i,j,w] for i in self.rrhs ) <= 1 for j in self.nodes for w in self.lambdas)#1

		#assures that the amount of sensors on a transmission channel does not exceed its capacity
		self.mdl.add_constraints(self.mdl.sum(self.x[i,j,w] * bandwidth_rate for i in self.sensors for j in self.nodes) <= wavelength_capacity[w] for w in self.lambdas)
		#assures that the amount of sensors being processed at node n does not exceed its processing capacity
		self.mdl.add_constraints(self.mdl.sum(self.x[i,j,w] * sensor_units for i in self.sensors for w in self.lambdas) <= node_capacity[j] for j in self.nodes)

		#activates a processing node n if the processing of sensors i is placed at node n
		self.mdl.add_constraints(B*self.xn[j] >= self.mdl.sum(self.x[i,j,w] for i in self.sensors for w in self.lambdas) for j in self.nodes)
		self.mdl.add_constraints(self.xn[j] <= self.mdl.sum(self.x[i,j,w] for i in self.sensors for w in self.lambdas) for j in self.nodes)
		#activates a transmission channel w at processing node n if sensors i is placed at node n and transmitted by w
		self.mdl.add_constraints(B*self.z[w,j] >= self.mdl.sum(self.x[i,j,w] for i in self.sensors) for w in self.lambdas for j in self.nodes)
		self.mdl.add_constraints(self.z[w,j] <= self.mdl.sum(self.x[i,j,w] for i in self.sensors) for w in self.lambdas for j in self.nodes)
		#Aassures that each transmission channel w is used to transmit only to node n - maybe we need to change it in our case to better fit our architecture
		self.mdl.add_constraints(self.mdl.sum(self.z[w,j] for j in self.nodes) <= 1 for w in self.lambdas)
		self.mdl.add_constraints(self.mdl.sum(self.y[i,j] for j in self.nodes) == 1 for i in self.sensors)
		#assures that node n is activated when sensor i is processed at it -similar to a previous constraint, but we need this if we want to 
		#extend this model to deal with virtual machines placement inside the node
		self.mdl.add_constraints(B*self.y[i,j] >= self.mdl.sum(self.x[i,j,w] for w in self.lambdas) for i in self.sensors for j in self.nodes)
		self.mdl.add_constraints(self.y[i,j] <= self.mdl.sum(self.x[i,j,w] for w in self.lambdas) for i in self.sensors  for j in self.nodes)

		
		#this constraints guarantees that each sensor can be allocated to either the cloud or to a fog node connected to it by a physical link
		self.mdl.add_constraints(self.y[i,j] <= self.fog[i][j] for i in self.sensors for j in self.nodes)
		self.mdl.add_constraints(self.z[w,j] <= lambda_node[w][j] for w in self.lambdas for j in self.nodes)

	#add new constraint - if we want to modify the model at run time - not working yet
	def addNewConstraint(self, constraint):
		self.mdl.add_constraints(constraint)

	#set the objective function - this example reduces the power consumption, i.e. activates the least number of processing nodes
	def setObjective(self):
		#self.mdl.minimize(self.mdl.sum(self.xn[j] * nodeCost[j] for j in self.nodes))
		self.mdl.minimize(self.mdl.sum(self.xn[j] * nodeCost[j] for j in self.nodes))

	#solves the model
	def solveILP(self):
		self.sol = self.mdl.solve()
		return self.sol

	#print variables values
	def print_var_values(self):
		for i in self.x:
			if self.x[i].solution_value >= 1:
				print("{} is {}".format(self.x[i], self.x[i].solution_value))
		for i in self.xn:
			if self.xn[i].solution_value >= 1:
				print("{} is {}".format(self.xn[i], self.xn[i].solution_value))

		for i in self.z:
			if self.z[i].solution_value >= 1:
				print("{} is {}".format(self.z[i], self.z[i].solution_value))


#return the variables values - this is the placement solution
	def return_solution_values(self):
		self.var_x = []
		self.var_xn = []
		self.var_z = []
		for i in self.x:
			if self.x[i].solution_value >= 1:
				self.var_x.append(i)
		for i in self.xn:
			if self.xn[i].solution_value >= 1:
				self.var_xn.append(i)
		for i in self.z:
			if self.z[i].solution_value >= 1:
				self.var_z.append(i)

		solution = Solution(self.var_x, self.var_xn, self.var_z)

		return solution



	#this method updates the network state based on the result of the ILP solution
	#it takes the node activated and updates its costs, the lambda allocated and the DUs capacity, either activate or not the switch
	#and also updates the cost and capacity of the lambda used
	#just remembering, when a lambda is allocated to its node, if this node is not being processed by the ilp, all lambdas allcoated
	#to it receives capacity 0 to guarantee that they will not be used
	#when both a node and one of its DUs are allocated, they costs are updated to 0 to guarantee that they are already activated 
	#when they are passed to be either or not selected to a new RRH, thus guaranteeing that they are already turned on and no additional
	#"turning on" cost will be computed
	#Finally, the updated made by this method only acts upon the activated node (and its DUs) and the allocated lambda




	def updateValues(self, solution):
		self.updateSensor(solution)
		#search the node(s) returned from the solution
		for key in solution.var_x:
			node_id = key[1]
			sensors_on_nodes[node_id] += 1
			node_capacity[node_id] -= sensor_units
			#node = pns[node_id]
			if nodeState[node_id] == 0:
				#not activated, updates costs
				nodeCost[node_id] = 0
				nodeState[node_id] = 1
			lambda_id = key[2]
			if lambda_state[lambda_id] == 0:
				lambda_state[lambda_id] = 1
				lc_cost[lambda_id] = 0
				ln = lambda_node[lambda_id]
				for i in range(len(ln)):
					if i == node_id:
						ln[i] = 1
					else:
						ln[i] = 0
			wavelength_capacity[lambda_id] -= bandwidth_rate	
		

	#put the solution values into the RRH
	def updateSensor(self,solution):
			for i in range(len(self.sensor)):
				self.sensor[i].var_x = solution.var_x[i]

	#deallocates the Sensors from the processing nodes and transmissions channels
	#This method takes the sensor to be deallocated and free the resources from the
	#data structures of the node, lambda, du and switches
	def deallocateSensor(self, rrh):
		#take the decision variables on the sensor and release the resources
		#take the node and the transmission channel
		node_id = sensor.var_x[1]
		sensors_on_nodes[node_id] -= 1
		node_capacity[node_id] += sensor_units
		lambda_id = sensor.var_x[2]
		du = sensor.var_u[2]
		#find the wavelength
		wavelength_capacity[lambda_id] += bandwidth_rate
		#now, updates the state and costs of the resources, if they were completely released
		if wavelength_capacity[lambda_id] == 10000.0 and lambda_state[lambda_id] == 1:
			lambda_state[lambda_id] = 0
			lc_cost[lambda_id] = 20.0
			for i in range(len(lambda_node[lambda_id])):
				lambda_node[lambda_id][i] = 1
		#check if the node has sensors being processed
		if sensors_on_nodes[node_id] == 0 and nodeState[node_id] == 1:
			nodeState[node_id] = 0
			if node_id == 0:
				nodeCost[node_id] = 600.0
			else:
				nodeCost[node_id] = 500.0


#encapsulates the solution values
class Solution(object):
	def __init__(self, var_x, var_xn, var_z):
		self.var_x = var_x
		self.var_xn = var_xn
		self.var_z = var_z

#this class represents a sensor containing its possible processing nodes
class Sensor(object):
	def __init__(self, aId, sensor_matrix):
		self.id = aId
		self.sensor_matrix = sensor_matrix
		self.var_x = None

#Utility class
class Util(object):

	#gets the overall delay of the network
	def overallDelay(self, solution):
		total_delay = 0.0
		for key in solution.var_xn:
			#print(key)
			if key == 0:
				total_delay = cloud_delay
			else:
				total_delay += fog_delay
		return (total_delay/len(solution.var_xn))


	#compute the power consumption at the moment, considering activated processing nodes and transmission channels
	def getPowerConsumption(self):
		netCost = 0.0
		#compute all activated nodes
		for i in range(len(nodeState)):
			if nodeState[i] == 1:
				if i == 0:
					netCost += 600.0
				else:
					netCost += 300.0
		#compute lambda and switch costs
		for w in lambda_state:
			if w == 1:
				netCost += 20.0
		return netCost

	#------------------------------------------------------------------------------------------#
	#--------------------------Methods for the dynamic case-------------------------------------#
	#create a list of sensors with its own connected processing nodes - for the dynamic case
	def newCreateSensors(self, amount):
		sensors = []
		for i in range(amount):
			r = Sensor(i, [1,0,0,0,0,0])
			sensors.append(r)
		self.setMatrix(sensors)
		return sensors

	#set the sensor_matrix for each sensor created - for the dynamic case - The sensor_matrix represents the physical connections between the sensor and the fog nodes
	def setMatrix(self, sensors):
		count = 1
		for r in sensors:
			if count <= len(r.sensor_matrix)-1:
				r.sensor_matrix[count] = 1
				count += 1
			else:
				count = 1
				r.sensor_matrix[count] = 1
				count += 1

	#------------------------------------------------------------------------------------------#
	#--------------------------Methods for the static case-------------------------------------#

	#set matrix 
	def staticSetMatrix(self, sensors, bottom, top, fog):
		for i in range(bottom, top):
			sensors[i].sensor_matrix[fog] = 1

	#another create sensors method to be used on the static case
	#it creates the sensors-fog matrix according to the amount of processing nodes declared
	def staticCreateSensors(self, sensors_amount):	
		#create the rrhs
		sensors = []
		for i in range(sensors_amount):
			r = Sensor(i, [1,0,0,0,0,0])
			sensors.append(r)
		return sensors

	#this method connects sensors uniformly to the fog nodes (example, 5 antenas, 5 fog nodes, 1 sensor connected per fog; 10 sensors, 5 fog, 2 sensors per fog and so on)
	def setExperiment(self, sensors, fogs):
		divided = int(len(sensors)/fogs)
		self.staticSetMatrix(sensors, 0, divided, 1)
		self.staticSetMatrix(sensors,  divided, divided*2, 2)
		self.staticSetMatrix(sensors,  divided*2, divided*3, 3)
		self.staticSetMatrix(sensors,  divided*3, divided*4, 4)
		self.staticSetMatrix(sensors,  divided*4, divided*5, 5)


#Testing
#creates utility class
util = Util()
#set lambdas
setLambdas()
#amount of sensors
amount = 45
#list to keep the sensors
sensors = []
#creates the sensors
sensors = util.newCreateSensors(amount)
ilp = ILP(sensors, range(len(sensors)), nodes, lambdas)
s = ilp.run()
solution = ilp.return_solution_values()
ilp.updateValues(solution)
print(util.getPowerConsumption())
print(s.solve_details.time)
