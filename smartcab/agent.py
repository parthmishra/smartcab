import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
	"""An agent that learns to drive in the smartcab world."""

	def __init__(self, env):
		super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
		self.color = 'red'  # override color
		self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
		# TODO: Initialize any additional variables here
		self.state = None
		self.all_actions = self.env.valid_actions
		self.prev_reward = None
		self.prev_action = None
		self.prev_state = None
		self.epsilon = 0.5
		self.alpha = 0.7
		self.gamma = 0.1
		self.time_step = 1.0
		self.successes = []
		self.Q_table = {}



	def reset(self, destination=None):
		self.planner.route_to(destination)
		# TODO: Prepare for a new trip; reset any variables here, if required
		self.prev_state = None
		self.prev_action = None
		self.prev_reward = None
		self.epsilon = 0.5
		self.alpha = 0.7
		self.gamma = 0.1
		self.time_step = 1.0
		self.successes.append(0)



	def update(self, t):
		# Gather inputs
		self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
		inputs = self.env.sense(self)
		inputs['next_waypoint'] = self.next_waypoint
		deadline = self.env.get_deadline(self)

		# TODO: Update state
		self.state = inputs
		self.state = tuple(self.state.items())
		# TODO: Select action according to your policy

		action = None # default action
	 
		self.epsilon = self.decayRate(self.time_step) # decay epsilon as time goes on there is less randomness
		
		# occasionally take a random action instaed of best Q, the chance of this decreases as epsilon gets smaller
		# from the decay function above
		if random.random() < self.epsilon: # rand value from [0,1] to compare against epsilon (also from [0,1])
			action = random.choice(self.all_actions)
			best_Q = self.setqValue(self.state, action)
		else:   
			# find state, action pair that returns highest q-value
			# NOTE: not necessary to gather all states
			best_Q = -1000.0
			for a in self.all_actions:
		
				Q = self.setqValue(self.state, a)

				if Q > best_Q: #
					best_Q = Q
					action = a

		# Execute action and get reward
		reward = self.env.act(self, action)
		self.time_step += 1.0

		# TODO: Learn policy based on state, action, reward


		if self.prev_state != None: # make sure at least 1 time step is completed b/c need prev state (current) and along with next state
			self.setqValue(self.prev_state, self.prev_action)
			utility = self.prev_reward + (self.gamma * best_Q)
			old_pair = (self.prev_state, self.prev_action)
			# create policy using q-learning function: Q(s,a) <- (1-learning rate)*old Q + learning rate*[reward + discount * best Q]
			self.Q_table[old_pair] = ( (1 - self.alpha) * self.Q_table[old_pair] ) + self.alpha * utility



		#print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
		self.prev_state = self.state
		self.prev_action = action
		self.prev_reward = reward

		# calculate success rate, credit forum user @ronrest
		location = self.env.agent_states[self]["location"] 
		destination = self.env.agent_states[self]["destination"]
		if location == destination:
			self.successes.pop()
			self.successes.append(1)

		num_success = self.successes.count(1)
		
		success_rate = (float(num_success)/float(len(self.successes)))*100

		print "Success Rate: {}%".format(success_rate)

	def decayRate(self, time_step): 
		return 1.0 / float(time_step)

	def setqValue(self, state, action):

		if (state, action) not in self.Q_table:
			self.Q_table[(state,action)] = 1.0 #initialize first visit S,A pair q values to 1
		return self.Q_table[(state,action)]






def run():
	"""Run the agent for a finite number of trials."""

	# Set up environment and agent
	e = Environment()  # create environment (also adds some dummy traffic)
	a = e.create_agent(LearningAgent)  # create agent
	e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
	# NOTE: You can set enforce_deadline=False while debugging to allow longer trials

	# Now simulate it
	sim = Simulator(e, update_delay=0.00001, display=False)  # create simulator (uses pygame when display=True, if available)
	# NOTE: To speed up simulation, reduce update_delay and/or set display=False

	sim.run(n_trials=100)  # run for a specified number of trials
	# NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
	run()
