import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import time
import random
import csv
import tensorflow.experimental.numpy as tfn
from tensorflow.keras import backend as K

StartingReplaySize = 2000
ExplorationMax = 1.0
MemorySize = 900000
TrainingFrequency = 4
AirlBatchSize = 1024
BatchSize = 128
Gamma = 0.999
ExplorationMin = 0.1
ExplorationSteps = 850000
ExplorationDecay = (ExplorationMax - ExplorationMin)/ExplorationSteps
ModelPersistenceUpdateFrequency = 10000
TargetUpdateFrequency = 5000


class Discriminator:
    def __init__(self, obs_shape, action_space, model_path):
        self.model_path = model_path
        inputs = keras.Input(shape=(obs_shape+action_space,), name="obs")
        x1 = layers.Dense(64, activation="relu")(inputs)
        x2 = layers.Dense(64, activation="relu")(x1)
        outputs = layers.Dense(1, name="predictions")(x2)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        if os.path.isdir("./discriminator"):
            if len(os.listdir("./discriminator")) != 0:
                self.model.load_weights(model_path)
        else:
            os.mkdir("./discriminator")


class Generator:
    def __init__(self, obs_shape, action_space, model_path):
        self.model_path = model_path
        inputs = keras.Input(shape=(obs_shape,), name="obs")
        x = layers.Dense(64)(inputs)
        x = layers.LeakyReLU(alpha=0.3)(x)
        x = layers.Dense(64)(x)
        x = layers.LeakyReLU(alpha=0.3)(x)
        x = layers.Dense(action_space, name="predictions", activation="sigmoid")(x)
        x = layers.Concatenate(axis=-1)([inputs, x])
        outputs = layers.Flatten()(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        if os.path.isdir("./generator"):
            if len(os.listdir("./generator")) != 0:
                self.model.load_weights(model_path)
        else:
            os.mkdir("./generator")

class Model:
    def __init__(self, obs_shape, action_space, expert):
        self.gmodel_path = "./generator/model"
        self.dmodel_path = "./discriminator/model"
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.discriminator = Discriminator(self.obs_shape, self.action_space, self.dmodel_path).model
        self.generator = Generator(self.obs_shape, self.action_space, self.gmodel_path).model
        self.epsilon = ExplorationMax
        self.memory = []
        for i in range(len(expert)):
            expert[i][0] = self.format(expert[i][0])
        self.expert = expert

    def extract(self, input_arr):
        return self.format(list(zip(*np.nonzero(input_arr))))

    def format(self, input_arr):
        input_arr = np.array(input_arr).flatten()
        if len(input_arr) < 25:
            print("padding...")
            input_arr = np.pad(input_arr, (0, 25 - len(input_arr)), mode="constant", constant_values=0)
            return input_arr
        if len(input_arr) > 25:
            print("clipping...")
            input_arr = input_arr[:24]
            return input_arr
        return input_arr

    def move(self, state):
        if np.random.rand() < self.epsilon or len(self.memory) < StartingReplaySize:
            print("Move Is Random")
            return random.randrange(self.action_space)
        print("Move Is Predicted")
        return np.argmax(self.generator.predict(np.expand_dims(np.asarray(self.extract(state)).astype(np.float64), axis=0), batch_size=1))

    def remember(self, current_state, action, reward, next_state):
        self.memory.append({"current_state": self.extract(current_state),
                            "action": action,
                            "reward": reward,
                            "next_state": self.extract(next_state)})
        if len(self.memory) > MemorySize:
            self.memory.pop(0)

    def fit(self, exp_obs, exp_act, obs, act, rews):
        for i in [exp_obs, exp_act, obs, act, rews]:
            print(np.asarray(i).shape)
        d_optimizer = keras.optimizers.Adam(learning_rate=0.0003)
        g_optimizer = keras.optimizers.Adam(learning_rate=0.0004)
        loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        epochs = 5
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()
            step = 0
            for eobs, eact, bobs, bact, rew in zip(exp_obs, exp_act, obs, act, rews):
                step += 1
                expexp = np.concatenate([eobs, [random.uniform(0, 0.9) if i != eact else 1 for i in range(19)]])
                expbot = np.append(eobs, [random.uniform(0, 0.9) if i != random.choice([j for j in range(self.action_space) if j not in [eact]]) else 1 for i in range(19)])
                bact = [random.uniform(0, 0.9) if i != eact else 1 for i in range(19)]
                botbot = np.append(bobs, bact)
                dataset = [expexp, expbot, botbot]
                dataset = tfn.asarray(dataset, dtype=tf.float32)
                labels = tf.expand_dims(tf.concat((tf.zeros((1,)), tf.ones((2,))), axis=0), axis=-1)
                labels += 0.05 * tf.random.uniform(labels.shape)
                print(dataset/tf.norm(dataset))
                with tf.GradientTape() as tape:
                    predictions = self.discriminator(dataset/tf.norm(dataset))
                    print(predictions)
                    d_loss = loss_fn(labels, predictions)
                grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
                d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
                misleading_labels = tf.zeros((1,))

                with tf.GradientTape() as tape:
                    gdataset = tf.expand_dims(tf.cast(tf.squeeze(self.generator(bobs)), tf.float32), axis=0)
                    predictions = self.discriminator(gdataset/tf.norm(gdataset))
                    g_loss = loss_fn(misleading_labels, predictions)-rew
                grads = tape.gradient(g_loss, self.generator.trainable_weights)
                g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

                if step % 32 == 0:
                    print("discriminator loss at step %d: %.2f" % (step, d_loss))
                    print("generator loss at step %d: %2.f" % (step, g_loss))
            print("Time taken: %.2fs" % (time.time() - start_time))
        return g_loss, d_loss

    def train(self):
        expert_trajectories = list(zip(*random.sample(self.expert, BatchSize)))
        batch = random.sample(self.memory, BatchSize)
        current_states = []
        actions = []
        rewards = []
        for entry in batch:
            current_states.append(np.expand_dims(np.asarray(entry["current_state"]).astype(np.float64), axis=0))
            actions.append(entry["action"])
            rewards.append(entry["reward"])
        gloss, dloss = self.fit(expert_trajectories[0], expert_trajectories[1], current_states, actions, rewards)
        return gloss, dloss

    def update_epsilon(self):
        self.epsilon -= ExplorationDecay
        self.epsilon = max(ExplorationMin, self.epsilon)

    def step_update(self, totalStep):
        if len(self.memory) < StartingReplaySize:
            return
        if totalStep % TrainingFrequency == 0:
            g_loss, d_loss = self.train()
            self.save_csv(path="gloss.csv", score=g_loss)
            self.save_csv(path="dloss.csv", score=d_loss)
        self.update_epsilon()
        if totalStep % ModelPersistenceUpdateFrequency == 0:
            self.generator.save_weights(self.gmodel_path)
            self.discriminator.save_weights(self.dmodel_path)
        if totalStep % TargetUpdateFrequency == 0:
            print('{{"metric": "epsilon", "value": {}}}'.format(self.epsilon))
            print('{{"metric": "total_step", "value": {}}}'.format(totalStep))

    def save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])
