import numpy as np

# Adapted from https://github.com/YuriyGuts/cartpole-q-learning/blob/master/cartpole.py

class QAgent:
    def __init__(self,
                 learning_rate=0.2,
                 discount_factor=1.0,
                 exploration_rate=0.5,
                 exploration_decay_rate=0.99,
                 num_of_actions=2,
                 max_episodes_to_run=5000,
                 max_timesteps_per_episode=200,
                 goal_avg_episode_length=195,
                 goal_consecutive_episodes=100):
        
        # agent
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.state = None
        self.action = None
        # episode
        self.max_episodes_to_run = max_episodes_to_run
        self.history = np.zeros(self.max_episodes_to_run, dtype=int)
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.goal_avg_episode_length = goal_avg_episode_length
        self.goal_consecutive_episodes = goal_consecutive_episodes

        self._state_bins = self.get_state_bins()
        # Create a clean Q-Table.
        self._num_actions = 2
        self._max_bins = max(len(bin) for bin in self._state_bins)
        num_states = (self._max_bins + 1) ** len(self._state_bins)
        self.q = np.zeros(shape=(num_states, self._num_actions))
    
    def get_state_bins():
        raise NotImplementedError

    def get_environment():
        raise NotImplementedError

    @staticmethod
    def _discretize_range(lower_bound, upper_bound, num_bins):
        return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

    @staticmethod
    def _discretize_value(value, bins):
        return np.digitize(x=value, bins=bins)

    def _build_state(self, observation):
        # Discretize the observation features and reduce them to a single integer.
        state = sum(
            self._discretize_value(feature, self._state_bins[i]) * ((self._max_bins + 1) ** i)
            for i, feature in enumerate(observation)
        )
        return state

    def begin_episode(self, observation):
        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_decay_rate

        # Get the action for the initial state.
        self.state = self._build_state(observation)
        return np.argmax(self.q[self.state])

    def act(self, observation, reward, learning=False):
        next_state = self._build_state(observation)

        # Exploration/exploitation: choose a random action or select the best one.
        enable_exploration = (1 - self.exploration_rate) <= np.random.uniform(0, 1)
        if enable_exploration and learning:
            next_action = np.random.randint(0, self._num_actions)
        else:
            next_action = np.argmax(self.q[next_state])

        # Learn: update Q-Table based on current reward and future action.
        if learning:
            self.q[self.state, self.action] += self.learning_rate * \
                (reward + self.discount_factor * max(self.q[next_state, :]) - self.q[self.state, self.action])

        self.state = next_state
        self.action = next_action
        return next_action

    def is_goal_reached(self, episode_index):
        avg = np.average(self.history[episode_index - self.goal_consecutive_episodes + 1:episode_index + 1])
        return avg >= self.goal_avg_episode_length

    def train(self, verbose=False):
        env = self.get_environment()
        for episode_index in range(self.max_episodes_to_run):
            observation = env.reset()
            action = self.begin_episode(observation)

            for timestep_index in range(self.max_timesteps_per_episode):
                # Perform the action and observe the new state.
                observation, reward, done, info = env.step(action)

                # If the episode has ended prematurely, penalize the agent.
                if done and timestep_index < self.max_timesteps_per_episode - 1:
                    reward = -self.max_episodes_to_run
                
                # Get the next action from the agent, given our new state (observation vector).
                action = self.act(observation, reward, True)

                # Record this episode to the history and check if the goal has been reached.
                if done or timestep_index == self.max_timesteps_per_episode - 1:
                    self.history[episode_index] = timestep_index + 1
                    if self.is_goal_reached(episode_index):
                        if verbose:
                            print(f'Goal reached after {episode_index + 1} episodes')
                        return (True, episode_index)
                    break
        if verbose:
            print(f'Goal NOT reached after {self.max_episodes_to_run} episodes')
        env.close()
        return (False, -1)

    def run(self, render=False):
        env = self.get_environment()
        frames = []
        observation = env.reset()
        action = self.begin_episode(observation)
        for timestep_index in range(self.max_timesteps_per_episode):
            observation, reward, done, info = env.step(action)
            if render:
                frames.append(env.render(mode = 'rgb_array'))
            episode_outcome = timestep_index == self.max_timesteps_per_episode - 1
            if done or episode_outcome:
                env.close()
                return (episode_outcome, frames)
            action = self.act(observation, reward)
        env.close()
        return (False, frames)
