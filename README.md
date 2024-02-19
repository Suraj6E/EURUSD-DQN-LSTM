# EURUSD-DQN-LSTM
Finding Optimal Day Trading Strategy in EURUSD using Deep Q-Networks (DQN) with Long Short-Term Memory (LSTM)

### ReInforment Learning
#### State, Action, Transition Modal, and Policy

* Action: Every possible Action
* State:  a learning agent learns, overtime, to behave optimally in a certain environment by interacting continuously in the environment. The agent during its course of learning experience various different situations in the environment it is in. These are called states. 

* Transition Modal: A Model (sometimes called Transition Model) gives an action’s effect in a state. In particular, T(S, a, S’) defines a transition T where being in state S and taking an action ‘a’ takes us to state S’ (S and S’ may be the same). For stochastic actions (noisy, non-deterministic) we also define a probability P(S’|S,a) which represents the probability of reaching a state S’ if action ‘a’ is taken in state S. Note Markov property states that the effects of an action taken in a state depend only on that state and not on the prior history. 
* Reward: A Reward is a real-valued reward function. R(s) indicates the reward for simply being in the state S. R(S,a) indicates the reward for being in a state S and taking an action ‘a’. R(S,a,S’) indicates the reward for being in a state S, taking an action ‘a’ and ending up in a state S’. 
* A Policy is a solution to the Markov Decision Process. A policy is a mapping from S to a. It indicates the action ‘a’ to be taken while in state S. 

### Q - Value

The Temporal Difference or TD-Update rule can be represented as follows :This update rule to estimate the value of Q is applied at every time step of the agents interaction with the environment. The terms used are explained below. :

![image.png](attachment:image.png)


S  : Current State of the agent.

A  : Current Action Picked according to some policy.

S'  : Next State where the agent ends up.

A'  : Next best action to be picked using current Q-value estimation, i.e. pick the 
action with the maximum Q-value in the next state.

R  : Current Reward observed from the environment in Response of current action.

$\gamma$  (>0 and <=1) : Discounting Factor for Future Rewards. Future rewards are less valuable than current rewards so they must be discounted. Since Q-value is an estimation of expected rewards from a state, discounting rule applies here as well.

$\alpha$  : Step length taken to update the estimation of Q(S, A).

### DQN
 Q-Learning creates an exact matrix for the working agent which it can “refer to” to maximize its reward in the long run. 
 To solve above problem, DQN is invented which uses a deep neural network to approximate the values. This approximation of values does not hurt as long as the relative importance is preserved. The basic working step for Deep Q-Learning is that the initial state is fed into the neural network and it returns the Q-value of all possible actions as an output. 

 ![image.png](attachment:image.png)

 

 Deep Q-Learning is a type of reinforcement learning algorithm that uses a deep neural network to approximate the Q-function, which is used to determine the optimal action to take in a given state

 One of the key challenges in implementing Deep Q-Learning is that the Q-function is typically non-linear and can have many local minima. This can make it difficult for the neural network to converge to the correct Q-function. To address this, several techniques have been proposed, such as experience replay and target networks.

Experience replay is a technique where the agent stores a subset of its experiences (state, action, reward, next state) in a memory buffer and samples from this buffer to update the Q-function. This helps to decorrelate the data and make the learning process more stable. Target networks, on the other hand, are used to stabilize the Q-function updates. In this technique, a separate network is used to compute the target Q-values, which are then used to update the Q-function network.

### Creating custom envionment using `gym`

`gym.Env.step(self, action: ActType) → Tuple[ObsType, float, bool, bool, dict]`
* Run one timestep of the environment’s dynamics.
* When end of episode is reached, you are responsible for calling `reset()` to reset this environment’s state. Accepts an action and returns either a tuple (observation, reward, terminated, truncated, info).



