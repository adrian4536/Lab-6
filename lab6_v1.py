import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration / Hyperparameters ---
NUM_BINS = (6, 12, 6, 12)  # discretization bins for each observation dimension
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 200

# --- Discretization bounds for CartPole observations ---
OBS_BOUNDS = [
    [-4.8, 4.8],          # cart position
    [-5, 5],              # cart velocity
    [-0.418, 0.418],      # pole angle (~24 degrees in radians)
    [-5, 5]               # pole velocity at tip
]

def discretize_state(state, bins):
    """Discretize continuous state into a tuple of bin indices."""
    discretized = []
    for i in range(len(state)):
        low, high = OBS_BOUNDS[i]
        clipped_state = np.clip(state[i], low, high)
        bin_idx = int(np.digitize(clipped_state, np.linspace(low, high, bins[i] - 1)))
        bin_idx = min(max(bin_idx, 0), bins[i] - 1)
        discretized.append(bin_idx)
    return tuple(discretized)

def initialize_q_table(bins):
    """Initialize Q-table with zeros."""
    q_shape = bins + (2,)
    return np.zeros(q_shape)

def choose_action(state, q_table, epsilon):
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(2)
    else:
        return np.argmax(q_table[state])

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    """Update the Q-table."""
    best_next_action = np.max(q_table[next_state])
    td_target = reward + gamma * best_next_action
    td_error = td_target - q_table[state][action]
    q_table[state][action] += alpha * td_error

def plot_training_results(rewards, steps):
    episodes = np.arange(len(rewards))
    
    plt.figure(figsize=(12, 5))

    # Plot total rewards
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, label="Reward per episode")
    plt.title("Reward over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)

    # Plot number of steps
    plt.subplot(1, 2, 2)
    plt.plot(episodes, steps, label="Steps per episode", color="orange")
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    env = gym.make("CartPole-v1")
    q_table = initialize_q_table(NUM_BINS)

    epsilon = EPSILON

    # Metrics tracking
    episode_rewards = []
    episode_steps = []

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        state_discrete = discretize_state(state, NUM_BINS)
        total_reward = 0
        steps = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            # Choose action
            action = choose_action(state_discrete, q_table, epsilon)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_discrete = discretize_state(next_state, NUM_BINS)

            # Update Q-table
            update_q_table(q_table, state_discrete, action, reward, next_state_discrete, LEARNING_RATE, DISCOUNT_FACTOR)

            state_discrete = next_state_discrete
            total_reward += reward
            steps += 1

            if done:
                break

        # Decay epsilon
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        # Record metrics
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        # Print detailed metrics every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            print(f"\nEpisode {episode + 1}/{NUM_EPISODES}")
            print(f"Average Reward (last 100): {avg_reward:.2f}")
            print(f"Average Steps (last 100): {avg_steps:.2f}")
            print(f"Epsilon: {epsilon:.3f}")

    #plot_training_results(episode_rewards, episode_steps)

    env.close()

    # Final summary
    print("\nTraining completed.")
    print(f"Average reward over {NUM_EPISODES} episodes: {np.mean(episode_rewards):.2f}")
    print(f"Average steps per episode: {np.mean(episode_steps):.2f}")

if __name__ == "__main__":
    main()
