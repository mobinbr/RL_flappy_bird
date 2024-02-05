import utils
import flappy_bird_gym
import random
import time
import numpy as np


class SmartFlappyBird:
    def __init__(self, iterations):
        self.Qvalues = utils.Counter()
        self.landa = 1
        self.epsilon = 0.8
        self.alpha = 0.5
        self.iterations = iterations

    def policy(self, state):
        return self.max_arg(state)

    @staticmethod
    def get_all_actions():
        # get two available actions, jump or not to
        return [0, 1]

    @staticmethod
    def convert_continuous_to_discrete(state):
        # return distance from center of next pipe vertically
        _, y = state
        simplified_y = round(y, 1)
        return 0, simplified_y

    def compute_reward(self, prev_info, new_info, done, observation):
        # compute reward of the action done
        reward = 0
        if done:
            reward = -1000

        else:
            # slight distance gets big reward
            if (0 <= observation[1] <= 0.05):
                new_info['score'] = prev_info['score'] + 1
                reward = 500

            # larger distance gets reward proportionally
            elif (observation[1] > 0.05):
                new_info['score'] = prev_info['score'] + 0.5
                reward = (1 / abs(observation[1]))

            # not much reward if bird is below the next pipe center
            elif (observation[1] < 0):
                reward = 1
        
        return reward

    def get_action(self, state):
        # change the normal distribution and return actionn
        # or use policy
        nuse_policy = utils.flip_coin(self.epsilon)
        if nuse_policy:
            random_number = random.randint(0, 100)
            actions = SmartFlappyBird.get_all_actions()
            if random_number < 90:
                return actions[0]
            else:
                return actions[1]
        else:
            return self.policy(state)

    def maxQ(self, state):
        # return maximum Q value of a state
        return max(self.Qvalues.get((state, action), 0) for action in self.get_all_actions())

    def max_arg(self, state):
        # return argument of the max Q of a state
        actions = self.get_all_actions()
        return actions[np.argmax([self.Qvalues.get((state, action), 0) for action in actions])]

    def update(self, reward, state, action, next_state):
        # update the q table using Q-learning formula
        max_a = self.max_arg(next_state)
        self.Qvalues[(state, action)] += self.alpha * (reward + self.landa * self.Qvalues[next_state, max_a] - self.Qvalues[(state, action)])

    def update_epsilon_alpha(self):
        # update epsilon and alpha exponentially
        if self.epsilon > 0.01:
            self.epsilon = self.epsilon * 0.95

        if self.alpha > 0.01:
            self.alpha = self.alpha * 0.95

    def run_with_policy(self, landa):
        # run the algorithm multiple times to train the model
        self.landa = landa
        env = flappy_bird_gym.make("FlappyBird-v0")
        observation = env.reset()
        info = {'score': 0}
        for _i in range(self.iterations):
            while True:
                observation = self.convert_continuous_to_discrete(tuple(observation))
                action = self.get_action(observation)  # policy affects here
                this_state = observation
                prev_info = info
                observation, reward, done, info = env.step(action)
                observation = self.convert_continuous_to_discrete(tuple(observation))
                this_state = self.convert_continuous_to_discrete(tuple(this_state))
                reward = self.compute_reward(prev_info, info, done, observation)
                self.update(reward, this_state, action, observation)
                self.update_epsilon_alpha()
                if done:
                    observation = env.reset()
                    break
        env.close()

    def run_with_no_policy(self, landa):
        # run the model based on trained model
        self.landa = landa
        self.alpha = 0
        self.epsilon = 0
        env = flappy_bird_gym.make("FlappyBird-v0")
        observation = env.reset()
        info = {'score': 0}
        while True:
            observation = self.convert_continuous_to_discrete(tuple(observation))
            action = self.get_action(observation)
            prev_info = info
            observation, reward, done, info = env.step(action)
            reward = self.compute_reward(prev_info, info, done, observation)
            env.render()
            time.sleep(1 / bird_speed)  # FPS
            if done:
                break
        env.close()

    def run(self):
        self.run_with_policy(1)
        self.run_with_no_policy(1)

# change bird speed (if you aren't patient enough)
bird_speed = 100 # FPS
# also change the iterations for training (patience problem again)
program = SmartFlappyBird(iterations=1000)
program.run()