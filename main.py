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
        return [0, 1]

    @staticmethod
    def convert_continuous_to_discrete(state):
        _, y = state
        simplified_y = round(y, 1)
        return 0, simplified_y

    def compute_reward(self, prev_info, new_info, done, observation):
        reward = 0
        if done:
            reward = -1000

        else:
            if (0 <= observation[1] <= 0.05):
                new_info['score'] = prev_info['score'] + 1
                reward = 500

            elif (observation[1] > 0.05):
                new_info['score'] = prev_info['score'] + 0.5
                reward = (1 / abs(observation[1]))

            elif (observation[1] < 0):
                reward = 1
        
        return reward

    def get_action(self, state):
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
        return max(self.Qvalues.get((state, action), 0) for action in self.get_all_actions())

    def max_arg(self, state):
        actions = self.get_all_actions()
        return actions[np.argmax([self.Qvalues.get((state, action), 0) for action in actions])]

    def update(self, reward, state, action, next_state):
        max_a = self.max_arg(next_state)
        self.Qvalues[(state, action)] += self.alpha * (reward + self.landa * self.Qvalues[next_state, max_a] - self.Qvalues[(state, action)])

    def update_epsilon_alpha(self):
        if self.epsilon > 0.01:
            self.epsilon = self.epsilon * 0.95

        if self.alpha > 0.01:
            self.alpha = self.alpha * 0.95

    def run_with_policy(self, landa):
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
        start = time.time()
        self.run_with_policy(1)
        end = time.time()
        print(end - start)
        self.run_with_no_policy(1)

bird_speed = 100 # FPS
program = SmartFlappyBird(iterations=1000)
program.run()