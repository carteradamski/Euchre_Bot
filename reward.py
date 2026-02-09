"""
reward.py  —  Reward functions for euchre DQN training.

Provides shaped rewards to guide learning:
- trick_reward: immediate feedback after each trick
- hand_reward: outcome-based reward at the end of each hand
"""


def trick_reward(my_team_won):
    """
    Reward for winning or losing a single trick.
    
    Args:
        my_team_won: bool, True if player's team won the trick
    
    Returns:
        +1 if won, -1 if lost
    """
    return 1.0 if my_team_won else -1.0


def hand_reward(team1_tricks, team1_calls, going_alone, alone_team1, my_team_is_team1):
    """
    Reward for the outcome of a complete hand.
    
    This captures the strategic value beyond individual tricks:
    - Winning the hand (3+ tricks)
    - Marching (all 5 tricks) 
    - Marching while going alone (4 points)
    - Getting euchred (losing when you called trump)
    
    Args:
        team1_tricks: int, number of tricks won by team 1 (0-5)
        team1_calls: bool, True if team 1 called trump
        going_alone: bool, True if someone went alone
        alone_team1: bool, True if the alone player is on team 1
        my_team_is_team1: bool, True if this player is on team 1
    
    Returns:
        float reward based on hand outcome
    """
    team1_won = (team1_tricks >= 3)
    my_team_won = (team1_won == my_team_is_team1)
    
    # Did my team call trump?
    my_team_called = (team1_calls == my_team_is_team1)
    
    # Base reward
    if not my_team_won:
        # Lost the hand
        if my_team_called:
            # Got euchred (called trump but lost)
            return -4.0
        else:
            # Lost but didn't call trump
            return -1.0
    
    # Won the hand
    tricks_won = team1_tricks if my_team_is_team1 else (5 - team1_tricks)
    
    if tricks_won == 5:
        # March (all 5 tricks)
        if going_alone and (alone_team1 == my_team_is_team1):
            # Marched while going alone (worth 4 game points)
            return 4.0
        else:
            # Regular march (worth 2 game points)
            return 2.0
    else:
        # Won 3-4 tricks (worth 1 game point)
        return 1.0
