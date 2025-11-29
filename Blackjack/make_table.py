import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


FILENAME = "agent_trained.json"

PLAYER_HARD_VALUES = range(4, 32) # A hard hand has no ace
PLAYER_SOFT_VALUES = range(12, 22) # A soft hand has an ace
DEALER_VALUES = range(1, 11)

def render_table(frame: pd.DataFrame, title: str):
    annotations = frame.map(lambda x: "stand" if x == 0 else "hit")

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        frame,
        annot=annotations,
        fmt="",
        cmap=["lightcoral", "lightgreen"],
        cbar=False,
    )
    plt.title(title)
    plt.xlabel("Dealer Showing")
    plt.ylabel("Player Total")
    plt.show()


with open(FILENAME, "r") as f:
    data: dict = json.load(f)


soft_frame = pd.DataFrame(
    np.zeros((len(PLAYER_SOFT_VALUES), len(DEALER_VALUES))),
    index = PLAYER_SOFT_VALUES,
    columns=DEALER_VALUES
)

hard_frame = pd.DataFrame(
    np.zeros((len(PLAYER_HARD_VALUES), len(DEALER_VALUES))),
    index = PLAYER_HARD_VALUES,
    columns=DEALER_VALUES
)

for key, weights in data.items():
    player, dealer, soft_hand = eval(key)
    stand, hit = weights

    if stand > hit:
        action = 0
    else:
        action = 1

    if soft_hand:
        soft_frame.loc[player, dealer] = action
    else:
        hard_frame.loc[player, dealer] = action

# Cleanup to make the frame render more nicely
hard_frame = hard_frame[hard_frame.index <= 21]
hard_frame = hard_frame.rename(columns={1: "A"})
cols = [c for c in hard_frame.columns if c != "A"] + ["A"]
hard_frame = hard_frame[cols]

soft_frame = soft_frame.rename(columns={1: "A"})
cols = [c for c in soft_frame.columns if c != "A"] + ["A"]
soft_frame = soft_frame[cols]

render_table(soft_frame, "Soft Hands (with ace)")
render_table(hard_frame, "Hard Hands (no ace)")
