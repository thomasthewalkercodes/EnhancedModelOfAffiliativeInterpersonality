# make everything
import sys

sys.path.append("..")
from Plotting import plot_probabilities
from game_engine import simulation
from different_games import get_type_game

l_rate = 0.5
e_rate = 3
rounds = 1000
t_game = "single"
detection_window = 10  # example value
reset_alpha = 0.1  # example value


if __name__ == "__main__":
    P1, P2 = get_type_game(t_game)

    result = simulation(P1, P2, l_rate, e_rate, rounds)
    plot_probabilities(result["p_history"], result["q_history"], t_game, None)
