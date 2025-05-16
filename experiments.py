def train_q_learning(env_name="CartPole-v1", bins=(6, 12, 6, 12), alpha=0.1, gamma=0.99,
                     epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, episodes=500, max_steps=200):
    env = gym.make(env_name)
    q_table = initialize_q_table(bins)
    rewards = []
    steps_list = []

    for ep in range(episodes):
        state, _ = env.reset()
        state_discrete = discretize_state(state, bins)
        total_reward = 0
        steps = 0

        for _ in range(max_steps):
            action = choose_action(state_discrete, q_table, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_discrete = discretize_state(next_state, bins)

            update_q_table(q_table, state_discrete, action, reward, next_state_discrete, alpha, gamma)
            state_discrete = next_state_discrete
            total_reward += reward
            steps += 1
            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards.append(total_reward)
        steps_list.append(steps)

    env.close()
    return rewards, steps_list

configs = [
    {"alpha": 0.1, "gamma": 0.95, "epsilon_decay": 0.99},
    {"alpha": 0.1, "gamma": 0.99, "epsilon_decay": 0.995},
    {"alpha": 0.05, "gamma": 0.99, "epsilon_decay": 0.990},
    {"alpha": 0.2, "gamma": 0.9, "epsilon_decay": 0.98}
]

for i, cfg in enumerate(configs):
    print(f"\n--- Running config {i+1}: {cfg} ---")
    rewards, steps = train_q_learning(
        alpha=cfg["alpha"],
        gamma=cfg["gamma"],
        epsilon_decay=cfg["epsilon_decay"],
        episodes=500
    )
    print(np.mean(rewards[-50:]))
    plot_training_results(rewards, steps)
