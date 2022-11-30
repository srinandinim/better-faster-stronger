from game.game import Game

def agent1(num_simulations, nodes=50):
    """
    run simulation n times and get statistics on success
    """
    agent_success = []
    timeouts = 0
    for _ in range(num_simulations):
        game = Game(nodes)
        game_success = game.run_agent_1()

        # agent caught the prey = 1, predator caught the agent/timeout = 0
        agent_success.append(1 if game_success == 1 else 0)

        # timeout if game_success returns -2
        timeouts = timeouts + 1 if game_success == -2 else timeouts

    wins = sum(agent_success)
    losses = len(agent_success) - wins - timeouts
    success = wins/(len(agent_success))
    print(f"Agent1: Wins: {wins}\tLosses: {losses}\tTimeouts: {timeouts}\tSuccess Rate: {round(success*100,2)}%")
    return wins, losses, timeouts, round(success*100, 2)
