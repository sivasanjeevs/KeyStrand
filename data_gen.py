"""
Keystrand: Keystroke Genomics & Forensic Hijack Detection
Module 1: Synthetic Data Generation & Calibration Collector

Generates:
1. calibration.csv: ~2400 labeled typing samples across 40 interrogation words
   for Person_A, Person_B, and Impostor in 'careful' and 'rushing'/'sleepy' states.
2. chat_history.json: 10 realistic messages in the group chat, including the 3 AM
   suspicious ransom message sent by the hijacked account.
"""

import json
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# Set reproducible seed
random.seed(42)
np.random.seed(42)

# Interrogation list: 40 frequent words in college/hostel chats
INTERROGATION_WORDS = [
    "biryani", "tomorrow", "tonight", "brother", "please",
    "account", "transfer", "hostel", "canteen", "library",
    "meeting", "money", "urgent", "hungry", "friend",
    "where", "coming", "already", "leaving", "arrived",
    "college", "project", "homework", "assignment", "message",
    "nothing", "problem", "seriously", "someone", "thinking",
    "talking", "dinner", "chicken", "swiggy", "zomato",
    "google", "number", "password", "thanks", "morning"
]

# QWERTY physical keyboard neighbor layout
QWERTY_NEIGHBORS = {
    'q': ['w', 'a', 's'],
    'w': ['q', 'e', 'a', 's', 'd'],
    'e': ['w', 'r', 's', 'd', 'f'],
    'r': ['e', 't', 'd', 'f', 'g'],
    't': ['r', 'y', 'f', 'g', 'h'],
    'y': ['t', 'u', 'g', 'h', 'j'],
    'u': ['y', 'i', 'h', 'j', 'k'],
    'i': ['u', 'o', 'j', 'k', 'l'],
    'o': ['i', 'p', 'k', 'l'],
    'p': ['o', 'l', '['],
    'a': ['q', 'w', 's', 'z'],
    's': ['a', 'w', 'e', 'd', 'x', 'z'],
    'd': ['s', 'e', 'r', 'f', 'c', 'x'],
    'f': ['d', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'],
    'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'j': ['h', 'u', 'i', 'k', 'm', 'n'],
    'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'],
    'z': ['a', 's', 'x'],
    'x': ['z', 's', 'd', 'c'],
    'c': ['x', 'd', 'f', 'v'],
    'v': ['c', 'f', 'g', 'b'],
    'b': ['v', 'g', 'h', 'n'],
    'n': ['b', 'h', 'j', 'm'],
    'm': ['n', 'j', 'k']
}

# Coordinate mapping for QWERTY keys (row, col)
KEYBOARD_COORDS = {
    'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4),
    'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9),
    'a': (1, 0.3), 's': (1, 1.3), 'd': (1, 2.3), 'f': (1, 3.3), 'g': (1, 4.3),
    'h': (1, 5.3), 'j': (1, 6.3), 'k': (1, 7.3), 'l': (1, 8.3),
    'z': (2, 0.7), 'x': (2, 1.7), 'c': (2, 2.7), 'v': (2, 3.7), 'b': (2, 4.7),
    'n': (2, 5.7), 'm': (2, 6.7)
}


def get_qwerty_distance(c1: str, c2: str) -> float:
    """Euclidean distance on a standard QWERTY keyboard."""
    c1 = c1.lower()
    c2 = c2.lower()
    if c1 == c2:
        return 0.0
    if c1 in KEYBOARD_COORDS and c2 in KEYBOARD_COORDS:
        r1, col1 = KEYBOARD_COORDS[c1]
        r2, col2 = KEYBOARD_COORDS[c2]
        return float(np.sqrt((r1 - r2) ** 2 + (col1 - col2) ** 2))
    return 3.5  # default max penalty


def simulate_typo_person_a(word: str, mood: str) -> str:
    """
    Person A profile:
    - Careful: Very accurate, rarely drops a key or hits an adjacent QWERTY key.
    - Rushing/Sleepy: Drops vowels aggressively ("tmrw", "plz", "briyani"),
      contracts words, occasionally transposes adjacent letters.
    """
    if mood == "careful":
        if random.random() < 0.88:
            return word
        # Mild typo: occasional adjacent key
        idx = random.randint(0, len(word) - 1)
        char = word[idx]
        if char in QWERTY_NEIGHBORS and random.random() < 0.7:
            sub = random.choice(QWERTY_NEIGHBORS[char])
            return word[:idx] + sub + word[idx + 1:]
        return word

    # Rushing / Sleepy
    chars = list(word)
    # Habit 1: Drop vowels (high probability)
    vowels = "aeiou"
    vowel_indices = [i for i, c in enumerate(chars) if c in vowels and i > 0]
    if vowel_indices and random.random() < 0.65:
        drop_idx = random.choice(vowel_indices)
        chars.pop(drop_idx)

    # Habit 2: Adjacent letter swap (transposition)
    if len(chars) > 3 and random.random() < 0.35:
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]

    # Habit 3: Specific common contractions
    result = "".join(chars)
    contractions = {
        "tomorrow": "tmrw", "tonight": "tnite", "brother": "bro",
        "please": "plz", "account": "acct", "transfer": "trfr",
        "message": "msg", "thanks": "thx", "biryani": "briyani"
    }
    if word in contractions and random.random() < 0.4:
        result = contractions[word]

    return result if result else word


def simulate_typo_person_b(word: str, mood: str) -> str:
    """
    Person B profile:
    - Careful: Methodical, almost zero spelling errors, accurate typing.
    - Rushing/Sleepy: Double-tap / held-down keys ("heeeello", "sooooon", "birryani"),
      trailing key repeats, affine gap insertion bursts.
    """
    if mood == "careful":
        if random.random() < 0.90:
            return word
        idx = random.randint(0, len(word) - 1)
        char = word[idx]
        if char in QWERTY_NEIGHBORS and random.random() < 0.5:
            return word[:idx] + random.choice(QWERTY_NEIGHBORS[char]) + word[idx + 1:]
        return word

    # Rushing / Sleepy: Double-tap or multi-tap habit (Affine insertion pattern)
    chars = list(word)
    if random.random() < 0.75 and len(chars) > 0:
        # Pick a target character to repeat (stuck key / fatigue)
        idx = random.randint(0, len(chars) - 1)
        repeat_count = random.choice([2, 3, 4])
        chars[idx] = chars[idx] * repeat_count

    # Also occasional neighbor substitution
    if random.random() < 0.3 and len(chars) > 1:
        idx = random.randint(0, len(chars) - 1)
        target = chars[idx][0]
        if target in QWERTY_NEIGHBORS:
            chars[idx] = random.choice(QWERTY_NEIGHBORS[target])

    return "".join(chars)


def simulate_typo_impostor(word: str, mood: str) -> str:
    """
    Impostor profile (Third person hijacking the account):
    - Completely different pattern: phonetic substitutions ("beriyani", "akount", "tumoro"),
      random deletion of word endings, hitting completely non-adjacent or distant keys,
      and lack of Person A's specific shorthand or Person B's double-taps.
    """
    phonetic_subs = {
        "biryani": "beriyani", "account": "akount", "tomorrow": "tumoro",
        "urgent": "urgnt", "please": "plzzz", "brother": "bhaiya",
        "hungry": "hungrry", "canteen": "kantin", "college": "colleg",
        "money": "mony", "google": "gpay", "dinner": "denner"
    }

    if random.random() < 0.45 and word in phonetic_subs:
        return phonetic_subs[word]

    chars = list(word)
    # Impostor has high substitution error rate and random edits
    edit_type = random.choice(["sub", "del", "ins", "noise"])
    if edit_type == "sub" and len(chars) > 0:
        idx = random.randint(0, len(chars) - 1)
        # Often hits random lowercase letter, not necessarily QWERTY neighbor
        chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz")
    elif edit_type == "del" and len(chars) > 3:
        idx = random.randint(0, len(chars) - 1)
        chars.pop(idx)
    elif edit_type == "ins":
        idx = random.randint(0, len(chars))
        chars.insert(idx, random.choice("aeiouxyz"))
    elif edit_type == "noise":
        chars.append(random.choice("!?12"))

    return "".join(chars)


def generate_calibration_dataset(
    words: List[str] = INTERROGATION_WORDS,
    reps_per_mood: int = 10
) -> pd.DataFrame:
    """
    Generates ~2400 labeled typing samples across 3 subjects and 2 moods,
    partitioned into train (70%), val (15%), test (15%).
    """
    records = []
    typists = [
        ("Person_A", simulate_typo_person_a),
        ("Person_B", simulate_typo_person_b),
        ("Impostor", simulate_typo_impostor)
    ]
    moods = ["careful", "rushing"]

    for typist_id, gen_func in typists:
        for mood in moods:
            for word in words:
                for rep in range(reps_per_mood):
                    typed = gen_func(word, mood)
                    records.append({
                        "typist_id": typist_id,
                        "intended_word": word,
                        "typed_word": typed,
                        "true_mood": mood,
                        "repetition": rep
                    })

    df = pd.DataFrame(records)

    # Stratified shuffle split: 70% train, 15% val, 15% test
    # Group by (typist_id, true_mood) to keep balanced distribution
    df["split"] = "train"
    for (t_id, m), group in df.groupby(["typist_id", "true_mood"]):
        indices = group.index.to_numpy().copy()
        np.random.shuffle(indices)
        n = len(indices)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        val_idx = indices[n_train: n_train + n_val]
        test_idx = indices[n_train + n_val:]

        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"

    return df


def generate_chat_history() -> List[Dict]:
    """
    Generates 10 realistic chat messages illustrating the hijacking scenario.
    Message #4 is the suspicious 3 AM emergency transfer request sent from Person_A's account by an Impostor!
    """
    messages = [
        {
            "id": 1,
            "timestamp": "2026-11-05 18:30:15",
            "sender_claimed": "Person_A",
            "actual_author": "Person_A",
            "mood": "careful",
            "text": "hey brother are you coming to library today for the assignment project",
            "note": "Legitimate study coordination from Person A."
        },
        {
            "id": 2,
            "timestamp": "2026-11-05 19:15:22",
            "sender_claimed": "Person_A",
            "actual_author": "Person_A",
            "mood": "rushing",
            "text": "leaving hostel now c u at canteen tmrw mrng",
            "note": "Person A rushing out; dropped vowels (tmrw, mrng)."
        },
        {
            "id": 3,
            "timestamp": "2026-11-05 21:40:02",
            "sender_claimed": "Person_A",
            "actual_author": "Person_A",
            "mood": "careful",
            "text": "did you submit the college homework to the portal already",
            "note": "Person A checking homework submission."
        },
        {
            "id": 4,
            "timestamp": "2026-11-06 03:02:44",
            "sender_claimed": "Person_A",
            "actual_author": "Impostor",
            "mood": "rushing",
            "text": "bhaiya send beriyani mony urgnt akount transfer right now",
            "note": "THE HIJACKED MESSAGE! 3 AM ransom demand showing Impostor's phonetic spelling habit (beriyani, mony, urgnt, akount)."
        },
        {
            "id": 5,
            "timestamp": "2026-11-06 03:05:18",
            "sender_claimed": "Person_B",
            "actual_author": "Person_B",
            "mood": "careful",
            "text": "wait who is this why are you asking money at 3am seriously",
            "note": "Person B suspicious of the sudden midnight request."
        },
        {
            "id": 6,
            "timestamp": "2026-11-06 03:07:05",
            "sender_claimed": "Person_A",
            "actual_author": "Impostor",
            "mood": "rushing",
            "text": "im so hungrry plzzz send to gpay fast no problem",
            "note": "Impostor pushing with non-standard typos (hungrry, plzzz, gpay)."
        },
        {
            "id": 7,
            "timestamp": "2026-11-06 08:14:50",
            "sender_claimed": "Person_A",
            "actual_author": "Person_A",
            "mood": "careful",
            "text": "what happened guys my phone was switched off all night in hostel",
            "note": "Real Person A wakes up confused."
        },
        {
            "id": 8,
            "timestamp": "2026-11-06 08:16:30",
            "sender_claimed": "Person_B",
            "actual_author": "Person_B",
            "mood": "rushing",
            "text": "soooomeone logged into your acct and asked for birryani cash",
            "note": "Person B typing fast with classic key-bounce repeats (soooomeone, birryani)."
        },
        {
            "id": 9,
            "timestamp": "2026-11-06 08:18:11",
            "sender_claimed": "Person_A",
            "actual_author": "Impostor",
            "mood": "rushing",
            "text": "beriyani swiggy order still waiting send mony brother",
            "note": "Impostor still trying one last attempt before session terminates."
        },
        {
            "id": 10,
            "timestamp": "2026-11-06 08:21:40",
            "sender_claimed": "Person_A",
            "actual_author": "Person_A",
            "mood": "careful",
            "text": "check the session logs immediately someone hijacked my account",
            "note": "Real Person A sounds the forensic alarm."
        }
    ]
    return messages


def build_and_save_data(csv_path: str = "calibration.csv", json_path: str = "chat_history.json"):
    """Builds and writes calibration.csv and chat_history.json to disk."""
    print("Generating calibration dataset...")
    df = generate_calibration_dataset()
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} samples to {csv_path}")

    print("Generating chat history dataset...")
    chats = generate_chat_history()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chats, f, indent=2)
    print(f"Saved {len(chats)} chat messages to {json_path}")
    return df, chats


if __name__ == "__main__":
    build_and_save_data()
