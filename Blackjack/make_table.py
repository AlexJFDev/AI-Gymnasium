import json
import pandas as pd
import numpy as np


FILENAME = "agent_trained.json"

PLAYER_NO_ACE_VALUES = range(4, 32)
PLAYER_ACE_VALUES = range(12, 22)
DEALER_VALUES = range(1, 11)


with open(FILENAME, "r") as f:
    data: dict = json.load(f)


ace_frame = pd.DataFrame(
    np.empty((len(PLAYER_ACE_VALUES), len(DEALER_VALUES)), dtype=str),
    index = PLAYER_ACE_VALUES,
    columns=DEALER_VALUES
)
ace_frame.index.name = "Player Hand Value"
ace_frame.columns.name = "Dealer Hand Value"

no_ace_frame = pd.DataFrame(
    np.empty((len(PLAYER_NO_ACE_VALUES), len(DEALER_VALUES)), dtype=str),
    index = PLAYER_NO_ACE_VALUES,
    columns=DEALER_VALUES
)
no_ace_frame.index.name = "Player Hand Value"
no_ace_frame.columns.name = "Dealer Hand Value"




for key, weights in data.items():
    player, dealer, player_has_ace = eval(key)
    stand, hit = weights

    if stand > hit:
        action = "stand"
    else:
        action = "hit"

    if player_has_ace:
        ace_frame.loc[player, dealer] = action
    else:
        no_ace_frame.loc[player, dealer] = action

print("Soft Hand (has an ace)")
print(ace_frame)
print()
print()
print()
print()
print("Hard Hand (no ace)")
print(no_ace_frame)
