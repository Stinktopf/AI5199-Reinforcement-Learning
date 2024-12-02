import numpy as np
import random
import ray
from ray import tune
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# --- Constants ---
GRID_SIZE = 11
NUM_TRAIN_EPISODES = 1001


# --- Q-Learning-Agent ---
class QLearningAgent:
    def __init__(self, grid_size, config, is_cat):
        self.pos = None
        self.grid_size = grid_size
        self.alpha = config["alpha"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon_min = config["epsilon_min"]
        self.max_steps = config["max_steps"]

        # Define possible actions per agent
        if is_cat:
            self.actions = ["up", "down", "left", "right", "stay"]
        else:
            self.actions = ["up", "down", "left", "right"]

        # Initialize random q-table (range[-0.01, 0.01]) for every x,y,action combination
        # Format: ((state), action) = quality of action for corresponding state (q-values)
        self.q_table = {
            ((x, y), action): random.uniform(-0.01, 0.01)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            for action in self.actions
        }

        # Initialize reward dictionary depending on agent type, cat or mouse
        self.rewards = config["cat_rewards"] if is_cat else config["mouse_rewards"]

    def reset(self, fixed_position):
        """Resets the position of the agent"""
        self.pos = fixed_position
        return self.pos

    def choose_action(self, state):
        """Choose an action based on epsilon (epsilon-greedy) or the q-value at the corresponding state."""

        # As epsilon gets lower the probability of random actions also get lower
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # Extract q-values at given state to take the max argument for action selection
        return self.actions[
            np.argmax([self.q_table.get((state, action)) for action in self.actions])
        ]

    def update_q(self, state, action, reward, next_state):
        """
        Adjusts the Q-value using the learning rate alpha, the received reward, the discount factor gamma, and the maximum future Q-value.
        This update moves the Q-value towards the estimated optimal value.
        """
        current_q = self.q_table.get((state, action))
        max_future_q = max(
            [self.q_table.get((next_state, a)) for a in self.actions], default=0
        )
        self.q_table[(state, action)] = current_q + self.alpha * (
            reward + self.gamma * max_future_q - current_q
        )

    def move(self, action):
        """
        Updates the agent's position in the grid based on the specified action and the current position.
        :param action: The action to be taken, which determines how the agent moves.
        :returns: A tuple containing:
                  - (x: int, y: int): The new position of the agent in the grid after the move.
                  - valid_move: A boolean indicating whether the action resulted in a valid move
        """

        x, y = self.pos

        # Directly return the last (current) position as a valid move
        if action == "stay":
            return self.pos, True

        # Update position of agent regarding the taken action
        if action == "up" and y > 0:
            y -= 1
        elif action == "down" and y < self.grid_size - 1:
            y += 1
        elif action == "left" and x > 0:
            x -= 1
        elif action == "right" and x < self.grid_size - 1:
            x += 1
        else:
            # If an invalid action was taken or the action would result that the agent moves outside the grid, return current position as an invalid move
            return self.pos, False

        # Return new position as valid move
        self.pos = (x, y)
        return self.pos, True


# --- Game Environment ---
class CatMouseGame:
    def __init__(self, config):
        self.grid_size = GRID_SIZE
        self.cat = QLearningAgent(GRID_SIZE, config, is_cat=True)
        self.mouse = QLearningAgent(GRID_SIZE, config, is_cat=False)
        self.cheese_positions = []

    def initialize_positions(self):
        """Initialize starting positions of agents and pieces of cheese."""
        self.mouse.reset(fixed_position=(5, 0))
        self.cat.reset(fixed_position=(5, 10))
        self.cheese_positions = [(0, 10), (10, 10)]

    @staticmethod
    def manhattan_dist(pos1, pos2):
        """Calculate the absolute manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def reward_agent(agent, agent_state, agent_action, reward, agent_next_state):
        agent.update_q(agent_state, agent_action, reward, agent_next_state)

    def train_episode(self):
        """
        Train one episode with the fixed starting positions and return if the episode was a success and the path the agents took.
        """

        cat_successes = 0
        mouse_successes = 0

        self.initialize_positions()
        cat_pos = self.cat.pos
        mouse_pos = self.mouse.pos
        path = [(cat_pos, mouse_pos, self.cheese_positions)]

        # Iterate over steps until max_steps or success was achieved
        for _ in range(self.cat.max_steps):
            # Choose action per agent depending on their state
            cat_action = self.cat.choose_action(cat_pos)
            mouse_action = self.mouse.choose_action(mouse_pos)

            # Move agents
            next_cat_pos, cat_valid_move = self.cat.move(cat_action)
            next_mouse_pos, mouse_valid_move = self.mouse.move(mouse_action)

            # Save position to path for later visualization
            path.append((next_cat_pos, next_mouse_pos, self.cheese_positions))

            # Initialize total reward
            total_cat_reward, total_mouse_reward = 0, 0

            # Cat agent didn't take a valid move -> penalize
            if not cat_valid_move:
                total_cat_reward += self.cat.rewards.get("out_of_bounds_penalty")

            # Mouse agent didn't take a valid move -> penalize
            if not mouse_valid_move:
                total_mouse_reward += self.mouse.rewards.get("out_of_bounds_penalty")

            # Calculate manhattan distance between mouse and cat
            old_agent_dist = self.manhattan_dist(cat_pos, mouse_pos)
            new_agent_dist = self.manhattan_dist(next_cat_pos, next_mouse_pos)

            # Calculate min manhattan distance between mouse and next cheese
            old_goal_dist = min(
                [
                    self.manhattan_dist(mouse_pos, cheese)
                    for cheese in self.cheese_positions
                ]
            )
            new_goal_dist = min(
                [
                    self.manhattan_dist(next_mouse_pos, cheese)
                    for cheese in self.cheese_positions
                ]
            )

            # Calculate rewards
            total_cat_reward += self.cat.rewards.get("distance_reward") * (
                old_agent_dist - new_agent_dist
            )
            total_mouse_reward += self.mouse.rewards.get("distance_reward") * (
                old_goal_dist - new_goal_dist
            )
            total_mouse_reward += self.mouse.rewards.get("avoid_cat_reward") * max(
                0, new_agent_dist - old_agent_dist
            )

            abort_episode = False

            # Mouse reached goal (cheese) -> Reward agent and update
            if next_mouse_pos in self.cheese_positions:
                total_mouse_reward += self.mouse.rewards.get("cheese_reward")
                mouse_successes = 1
                abort_episode = True
            elif (next_cat_pos == next_mouse_pos) or (
                next_cat_pos == mouse_pos and next_mouse_pos == cat_pos
            ):
                # Cat catches mouse by direct overlap and position switch -> Reward and penalize
                total_cat_reward += self.cat.rewards.get("catch_reward")
                total_mouse_reward += self.mouse.rewards.get("catch_penalty")
                cat_successes = 1
                abort_episode = True

            # Update q-values
            self.reward_agent(
                self.mouse, mouse_pos, mouse_action, total_mouse_reward, next_mouse_pos
            )
            self.reward_agent(
                self.cat, cat_pos, cat_action, total_cat_reward, next_cat_pos
            )

            # Update positions of agents
            cat_pos, mouse_pos = next_cat_pos, next_mouse_pos

            # Abort current episode if termination condition is met
            if abort_episode:
                break

        # Reduce epsilon
        self.cat.epsilon = max(
            self.cat.epsilon_min, self.cat.epsilon * self.cat.epsilon_decay
        )
        self.mouse.epsilon = max(
            self.mouse.epsilon_min, self.mouse.epsilon * self.mouse.epsilon_decay
        )

        return cat_successes, mouse_successes, path


# --- Ray Tune ---
ray.init(_temp_dir="C:/ray_temp", ignore_reinit_error=True)


def trial_dirname_creator(trial):
    return f"trial_{trial.trial_id}"


search_space = {
    "alpha": tune.uniform(0.01, 0.5),
    "gamma": tune.uniform(0.8, 0.99),
    "epsilon": tune.uniform(0.1, 0.8),
    "epsilon_decay": tune.uniform(0.99, 0.999),
    "epsilon_min": tune.uniform(0.01, 0.05),
    "max_steps": tune.randint(25, 100),
    "cat_rewards": tune.sample_from(
        lambda _: {
            "catch_reward": random.uniform(50, 100),
            "distance_reward": random.uniform(25, 50),
            "out_of_bounds_penalty": random.uniform(-10, -1),
        }
    ),
    "mouse_rewards": tune.sample_from(
        lambda _: {
            "cheese_reward": random.uniform(5, 100),
            "distance_reward": random.uniform(1, 50),
            "avoid_cat_reward": random.uniform(1, 50),
            "catch_penalty": random.uniform(-50, -10),
            "out_of_bounds_penalty": random.uniform(-100, -10),
        }
    ),
}


def train_cat_mouse(config):
    game = CatMouseGame(config)

    # // -> Floor division
    max_successes = NUM_TRAIN_EPISODES // 2

    cat_successes = 0
    mouse_successes = 0

    for _ in range(NUM_TRAIN_EPISODES):
        cat_successes_inner, mouse_successes_inner, _ = game.train_episode()
        cat_successes = min(cat_successes + cat_successes_inner, max_successes)
        mouse_successes = min(mouse_successes + mouse_successes_inner, max_successes)

    session.report({"successes": cat_successes + mouse_successes})


search_alg = OptunaSearch()
analysis = tune.run(
    train_cat_mouse,
    config=search_space,
    metric="successes",
    mode="max",
    num_samples=20,
    trial_dirname_creator=trial_dirname_creator,
)

# --- Best Configuration ---
best_config = analysis.best_config
print(f"Best Configuration: {best_config}")

# --- Animation ---
game = CatMouseGame(best_config)

# Further Training with the Best Configuration
print("Training the game with the best configuration...")

all_paths = []
cat_wins = []
mouse_wins = []

cat_successes = 0
mouse_successes = 0

for _ in range(NUM_TRAIN_EPISODES):
    cat_successes_inner, mouse_successes_inner, path = game.train_episode()
    all_paths.append(path)

    cat_wins.append(cat_successes_inner)
    mouse_wins.append(mouse_successes_inner)
    cat_successes += cat_successes_inner
    mouse_successes += mouse_successes_inner

print(f"Total Successful Episodes: {cat_successes + mouse_successes}")

# Select every n-th episode for animation
n = 100
selected_paths = all_paths[::n]

# Calculate the total number of frames
step_interval = 1
total_frames = sum(len(path) for path in selected_paths) * step_interval

# --- Enhanced Graphics Setup ---
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(3, 4, figure=fig)

# Game Field
ax_field = fig.add_subplot(gs[:, :3])
ax_field.set_xlim(-0.5, GRID_SIZE - 0.5)
ax_field.set_ylim(-0.5, GRID_SIZE - 0.5)
ax_field.set_xticks(range(GRID_SIZE))
ax_field.set_yticks(range(GRID_SIZE))
ax_field.grid(True, linestyle="--", linewidth=0.5, color="gray")
ax_field.set_title("Cat-Mouse Simulation", fontsize=16, fontweight="bold")

# Initialize Scatter Objects for Cat, Mouse, and Cheese
cat_scatter = ax_field.scatter([], [], color="#E63946", s=120, label="Cat")
mouse_scatter = ax_field.scatter([], [], color="#457B9D", s=120, label="Mouse")
cheese_scatter = ax_field.scatter(
    [], [], color="#F4A261", s=150, marker="*", label="Cheese"
)
ax_field.legend(loc="lower left", fontsize=12)

# --- Cumulative Wins ---
ax_wins = fig.add_subplot(gs[0, 3])
ax_wins.plot(
    range(1, len(cat_wins) + 1),
    np.cumsum(cat_wins),
    color="#E63946",
    label="Cat",
    lw=2.5,
)
ax_wins.plot(
    range(1, len(mouse_wins) + 1),
    np.cumsum(mouse_wins),
    color="#457B9D",
    label="Mouse",
    lw=2.5,
)
ax_wins.set_title("Cumulative Wins per Episode", fontsize=14, fontweight="bold")
ax_wins.set_xlabel("Episode", fontsize=12)
ax_wins.set_ylabel("Cumulative Wins", fontsize=12)
ax_wins.legend(loc="upper left", fontsize=10)
ax_wins.grid(True, linestyle="--", alpha=0.5)

# --- Heatmap for the Cat ---
cat_heatmap = np.zeros((GRID_SIZE, GRID_SIZE))
mouse_heatmap = np.zeros((GRID_SIZE, GRID_SIZE))

# Iterate through all episodes and count visited positions
for path in all_paths:
    for cat_pos, mouse_pos, _ in path:
        cat_heatmap[cat_pos[1], cat_pos[0]] += 1
        mouse_heatmap[mouse_pos[1], mouse_pos[0]] += 1

# Heatmap for the Cat
ax_cat_heatmap = fig.add_subplot(gs[1, 3])
cat_hm = ax_cat_heatmap.imshow(
    cat_heatmap, cmap="Reds", origin="lower", interpolation="nearest"
)

ax_cat_heatmap.set_title("Visited Positions (Cat)", fontsize=14, fontweight="bold")
ax_cat_heatmap.set_xlabel("X Position", fontsize=12)
ax_cat_heatmap.set_ylabel("Y Position", fontsize=12)
plt.colorbar(cat_hm, ax=ax_cat_heatmap, orientation="vertical", label="Visit Count")

# --- Heatmap for the Mouse ---
ax_mouse_heatmap = fig.add_subplot(gs[2, 3])
mouse_hm = ax_mouse_heatmap.imshow(
    mouse_heatmap, cmap="Blues", origin="lower", interpolation="nearest"
)

ax_mouse_heatmap.set_title("Visited Positions (Mouse)", fontsize=14, fontweight="bold")
ax_mouse_heatmap.set_xlabel("X Position", fontsize=12)
ax_mouse_heatmap.set_ylabel("Y Position", fontsize=12)
plt.colorbar(mouse_hm, ax=ax_mouse_heatmap, orientation="vertical", label="Visit Count")


# --- Animation Initialization ---
def init():
    """Initializes the animation and resets all values."""
    cat_scatter.set_offsets(np.empty((0, 2)))
    mouse_scatter.set_offsets(np.empty((0, 2)))
    cheese_scatter.set_offsets(np.empty((0, 2)))
    ax_field.set_title("Cat-Mouse Simulation", fontsize=16, fontweight="bold")
    return cat_scatter, mouse_scatter, cheese_scatter


# --- Animation ---
def animate(frame):
    """Updates the animation for the current frame from the selected paths."""
    cumulative_frames = 0
    step_size = n

    for episode_idx, path in enumerate(selected_paths):
        num_steps = len(path) * step_interval

        if cumulative_frames + num_steps > frame:
            step_idx = (frame - cumulative_frames) // step_interval
            if step_idx >= len(path):
                step_idx = len(path) - 1

            cat_pos, mouse_pos, cheese_positions = path[step_idx]

            cat_scatter.set_offsets([cat_pos])
            mouse_scatter.set_offsets([mouse_pos])
            cheese_scatter.set_offsets(cheese_positions)

            episode_number = episode_idx * step_size
            ax_field.set_title(
                f"Cat-Mouse Simulation - Episode {episode_number}/{NUM_TRAIN_EPISODES - 1}",
                fontsize=16,
                fontweight="bold",
            )

            return cat_scatter, mouse_scatter, cheese_scatter

        cumulative_frames += num_steps

    return cat_scatter, mouse_scatter, cheese_scatter


# Create Animation
ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=total_frames, interval=20, repeat=False
)

plt.tight_layout()
plt.show()
