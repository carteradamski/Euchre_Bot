from Card import card_to_index


def encode_state(game_state):
    """
    GameState  →  60-dim float32 tensor.

    Index map:
        [ 0– 23]  my_hand              24-dim binary,  1 if card is in my hand
        [24– 47]  cards_played_this_trick  24-dim binary,  1 if card already played this trick
        [48– 51]  trump suit           4-dim one-hot
        [52]      trick_index          scalar normalised to [0, 1]   (trick_index / 4)
        [53– 56]  my seat position     4-dim one-hot
        [57]      team1_tricks         scalar normalised  (/ 5)
        [58]      team2_tricks         scalar normalised  (/ 5)
        [59]      is_leading flag      1.0 if led_card is None (I am leading), else 0.0
    """
    import torch
    vec = torch.zeros(60, dtype=torch.float32)

    # [0-23] my hand
    for card in game_state.my_hand:
        vec[card_to_index(card)] = 1.0

    # [24-47] cards already played this trick (by anyone before me)
    for (_, card) in game_state.cards_played_this_trick:
        vec[24 + card_to_index(card)] = 1.0

    # [48-51] trump one-hot
    vec[48 + int(game_state.trump)] = 1.0

    # [52] trick index normalised
    vec[52] = game_state.trick_index / 4.0

    # [53-56] my seat one-hot
    vec[53 + game_state.my_index] = 1.0

    # [57-58] trick counts normalised
    vec[57] = game_state.team1_tricks / 5.0
    vec[58] = game_state.team2_tricks / 5.0

    # [59] am I leading?
    vec[59] = 1.0 if game_state.led_card is None else 0.0

    return vec


def build_legal_mask(game_state):
    """
    Returns a 24-dim binary tensor: 1.0 = legal action, 0.0 = illegal.

    Rules enforced:
        - Only cards actually in my_hand can be chosen.
        - If led_card is not None AND I have a card of the led suit, I must
          play one of those cards (follow-suit enforcement).
        - If led_card is None I am leading, so every card in hand is legal.
    """
    import torch
    mask = torch.zeros(24, dtype=torch.float32)

    if game_state.led_card is None:
        # Leading — anything in hand is fine
        for card in game_state.my_hand:
            mask[card_to_index(card)] = 1.0
    else:
        led_suit = game_state.led_card.get_suit(game_state.trump)

        # Check if I have any card of the led suit (trump-aware)
        has_led_suit = any(
            c.get_suit(game_state.trump) == led_suit
            for c in game_state.my_hand
        )

        for card in game_state.my_hand:
            if has_led_suit:
                # Must follow — only mark cards matching led suit
                if card.get_suit(game_state.trump) == led_suit:
                    mask[card_to_index(card)] = 1.0
            else:
                # Can't follow — anything goes
                mask[card_to_index(card)] = 1.0

    return mask
