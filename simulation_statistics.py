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
    print(
        f"Agent1: Wins: {wins}\tLosses: {losses}\tTimeouts: {timeouts}\tSuccess Rate: {round(success*100,2)}%")
    return wins, losses, timeouts, round(success*100, 2)


def agent2(num_simulations, nodes=50):
    """
    run simulation n times and get statistics on success
    """
    agent_success = []
    timeouts = 0
    for _ in range(num_simulations):
        game = Game(nodes)
        game_success = game.run_agent_2()

        # agent caught the prey = 1, predator caught the agent/timeout = 0
        agent_success.append(1 if game_success == 1 else 0)

        # timeout if game_success returns -2
        timeouts = timeouts + 1 if game_success == -2 else timeouts

    wins = sum(agent_success)
    losses = len(agent_success) - wins - timeouts
    success = wins/(len(agent_success))
    print(
        f"Agent2: Wins: {wins}\tLosses: {losses}\tTimeouts: {timeouts}\tSuccess Rate: {round(success*100,2)}%")
    return wins, losses, timeouts, round(success*100, 2)


def agent3(num_simulations, nodes=50):
    """
    run simulation n times and get statistics on success
    """
    agent_success = []
    timeouts = 0
    found_prey = 0
    for _ in range(num_simulations):
        game = Game(nodes)
        game_success, game_found_prey = game.run_agent_3()

        # agent caught the prey = 1, predator caught the agent/timeout = 0
        agent_success.append(1 if game_success == 1 else 0)

        # timeout if game_success returns -2
        timeouts = timeouts + 1 if game_success == -2 else timeouts

        # stores how often the agent knew where the prey was
        found_prey = found_prey + game_found_prey

    wins = sum(agent_success)
    losses = len(agent_success) - wins - timeouts
    success = wins/(len(agent_success))
    print(
        f"Agent3: Wins: {wins}\tLosses: {losses}\tTimeouts: {timeouts}\tSuccess Rate: {round(success*100,2)}%")
    return wins, losses, timeouts, round(success*100, 2), found_prey/num_simulations


def agent4(num_simulations, nodes=50):
    """
    run simulation n times and get statistics on success
    """
    agent_success = []
    timeouts = 0
    found_prey = 0
    for _ in range(num_simulations):
        game = Game(nodes)
        game_success, game_found_prey = game.run_agent_4()

        # agent caught the prey = 1, predator caught the agent/timeout = 0
        agent_success.append(1 if game_success == 1 else 0)

        # timeout if game_success returns -2
        timeouts = timeouts + 1 if game_success == -2 else timeouts

        # stores how often the agent knew where the prey was
        found_prey = found_prey + game_found_prey

    wins = sum(agent_success)
    losses = len(agent_success) - wins - timeouts
    success = wins/(len(agent_success))
    print(
        f"Agent4: Wins: {wins}\tLosses: {losses}\tTimeouts: {timeouts}\tSuccess Rate: {round(success*100,2)}%")
    return wins, losses, timeouts, round(success*100, 2), found_prey/num_simulations


def agent1rl(num_simulations, nodes=50):
    """
    run simulation n times and get statistics on success
    """
    agent_success = []
    timeouts = 0
    for _ in range(num_simulations):
        game = Game(nodes=nodes)
        game_success = game.run_agent_1_rl()

        # agent caught the prey = 1, predator caught the agent/timeout = 0
        agent_success.append(1 if game_success == 1 else 0)

        # timeout if game_success returns -2
        timeouts = timeouts + 1 if game_success == -2 else timeouts

    wins = sum(agent_success)
    losses = len(agent_success) - wins - timeouts
    success = wins/(len(agent_success))
    print(
        f"Agent1RL: Wins: {wins}\tLosses: {losses}\tTimeouts: {timeouts}\tSuccess Rate: {round(success*100,2)}%")
    return wins, losses, timeouts, round(success*100, 2)


def agent1rlnn(num_simulations, nodes=50):
    """
    run simulation n times and get statistics on success
    """
    agent_success = []
    timeouts = 0
    for _ in range(num_simulations):
        game = Game(nodes=nodes)
        game_success = game.run_agent_1_rl_nn()

        # agent caught the prey = 1, predator caught the agent/timeout = 0
        agent_success.append(1 if game_success == 1 else 0)

        # timeout if game_success returns -2
        timeouts = timeouts + 1 if game_success == -2 else timeouts

    wins = sum(agent_success)
    losses = len(agent_success) - wins - timeouts
    success = wins/(len(agent_success))
    print(
        f"Agent1RLNN: Wins: {wins}\tLosses: {losses}\tTimeouts: {timeouts}\tSuccess Rate: {round(success*100,2)}%")
    return wins, losses, timeouts, round(success*100, 2)


def agent3rl(num_simulations, nodes=50):
    """
    run simulation n times and get statistics on success
    """
    agent_success = []
    timeouts = 0
    for _ in range(num_simulations):
        game = Game(nodes=nodes)
        game_success = game.run_agent_3_rl()

        # agent caught the prey = 1, predator caught the agent/timeout = 0
        agent_success.append(1 if game_success == 1 else 0)

        # timeout if game_success returns -2
        timeouts = timeouts + 1 if game_success == -2 else timeouts

    wins = sum(agent_success)
    losses = len(agent_success) - wins - timeouts
    success = wins/(len(agent_success))
    print(
        f"Agent3RL: Wins: {wins}\tLosses: {losses}\tTimeouts: {timeouts}\tSuccess Rate: {round(success*100,2)}%")
    return wins, losses, timeouts, round(success*100, 2)


def agent3rlnn(num_simulations, nodes=50):
    """
    run simulation n times and get statistics on success
    """
    agent_success = []
    timeouts = 0
    for _ in range(num_simulations):
        game = Game(nodes=nodes)
        game_success = game.run_agent_3_rlnn()

        # agent caught the prey = 1, predator caught the agent/timeout = 0
        agent_success.append(1 if game_success == 1 else 0)

        # timeout if game_success returns -2
        timeouts = timeouts + 1 if game_success == -2 else timeouts

    wins = sum(agent_success)
    losses = len(agent_success) - wins - timeouts
    success = wins/(len(agent_success))
    print(
        f"Agent3RLNN: Wins: {wins}\tLosses: {losses}\tTimeouts: {timeouts}\tSuccess Rate: {round(success*100,2)}%")
    return wins, losses, timeouts, round(success*100, 2)
