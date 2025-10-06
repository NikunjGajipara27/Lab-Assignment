#!/usr/bin/env python3
"""
raag_bhairav_final.py

Generate melodies in North Indian Classical Raag Bhairav using a Genetic Algorithm.
Saves:
 - notation text file(s): bhairav_melody_seed<seed>.txt
 - MIDI file(s): bhairav_melody_seed<seed>.mid   (written via mido; no python-rtmidi required for file saving)
 - optional WAV via FluidSynth if you provide a SoundFont and fluidsynth is installed:
     fluidsynth -ni SoundFont.sf2 melody.mid -F melody.wav -r 44100

Dependencies:
 - Python 3.8+
 - numpy (pip install numpy)
 - matplotlib (pip install matplotlib) for the fitness plot
 - mido (pip install mido)  <--- required to write MIDI files
 - (optional) fluidsynth on system for MIDI->WAV rendering
   - On Windows: install FluidSynth binary or use MuseScore to export MIDI to WAV
   - On Ubuntu: sudo apt-get install fluidsynth

Usage:
    python raag_bhairav_final.py
    # To render WAV automatically (optional), set SOUNDFONT_PATH variable below.

"""

import os
import random
import math
from typing import List, Tuple
import argparse
import json

# External libs
try:
    import numpy as np
except Exception:
    print("Please install numpy: pip install numpy")
    raise

try:
    import matplotlib.pyplot as plt
except Exception:
    print("Please install matplotlib: pip install matplotlib")
    raise

# mido is required to write MIDI files. It's pure-Python for file creation.
try:
    import mido
    from mido import Message, MidiFile, MidiTrack, bpm2tempo
except Exception:
    mido = None

# Optional: subprocess for calling fluidsynth if available
import subprocess
import shutil
import time

# ----------------------------
# Configuration (tweak here)
# ----------------------------
OUTPUT_DIR = "bhairav_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GA params (tune for quality / runtime)
POP_SIZE = 300
GENERATIONS = 600
MELODY_LEN = 16
MUTATION_RATE = 0.18
CROSSOVER_RATE = 0.75
ELITISM = 6
TOURNAMENT_K = 3
SEED = 12345

# Musical params
SA_MIDI = 60  # choose Sa = MIDI 60 (C4)
TEMPO_BPM = 70
DURATION_QUARTER_NOTE = 1  # for MIDI writing (one quarter note per note)

# SoundFont path (optional) for fluidsynth rendering; set to None to skip
SOUNDFONT_PATH = None  # e.g., r"C:\sounds\GeneralUser GS v1.471.sf2"

# ----------------------------
# Raag Bhairav scale and motifs
# ----------------------------
NOTE_NAMES = ["S", "r", "G", "M", "P", "d", "N"]
SCALE_TO_SEMITONES = {"S":0,"r":1,"G":3,"M":5,"P":7,"d":8,"N":10}
ALLOWED_NOTES = NOTE_NAMES.copy()

# Representative motifs (pakad) to reward
PAKAD = [
    ("S","r","G"),
    ("r","G","M"),
    ("G","M","P"),
    ("P","d","N","S"),
    ("N","S","r"),
    ("r","G","M","P"),
    ("G","M","P","d")
]

# ----------------------------
# Utilities: genotype/phenotype
# ----------------------------
def random_melody(length: int) -> List[str]:
    return [random.choice(ALLOWED_NOTES) for _ in range(length)]

def melody_to_midi_notes(melody: List[str], sa_midi=SA_MIDI) -> List[int]:
    return [sa_midi + SCALE_TO_SEMITONES[n] for n in melody]

# ----------------------------
# Fitness function
# ----------------------------
def fitness(melody: List[str]) -> float:
    """Return higher-is-better fitness combining multiple musical heuristics."""
    score = 0.0
    L = len(melody)

    # Reward allowed notes (penalize foreign — should not occur)
    for n in melody:
        if n not in ALLOWED_NOTES:
            score -= 50.0

    # Motif matching reward
    for motif in PAKAD:
        mlen = len(motif)
        for i in range(L - mlen + 1):
            seg = tuple(melody[i:i+mlen])
            if seg == motif:
                score += 6.0
            else:
                # near-match reward (Hamming distance 1)
                if len(seg) == mlen:
                    hd = sum(1 for a,b in zip(seg, motif) if a != b)
                    if hd == 1:
                        score += 1.5

    # Stepwise motion reward / jump penalty
    midi = melody_to_midi_notes(melody)
    for i in range(1, L):
        jump = abs(midi[i] - midi[i-1])
        if jump <= 2:
            score += 1.0
        elif jump <= 4:
            score += 0.2
        else:
            score -= 0.8
        if jump >= 7:
            score -= 1.5

    # Encouraging mix of ups and downs
    ups = sum(1 for i in range(1,L) if midi[i]>midi[i-1])
    downs = sum(1 for i in range(1,L) if midi[i]<midi[i-1])
    score += 0.2 * min(ups, downs)

    # Ending preference
    if melody[-1] == "S":
        score += 4.0
    elif melody[-1] == "P":
        score += 2.0

    # Penalize monotony (long runs of same note)
    repeats = 0
    run = 1
    for i in range(1,L):
        if melody[i] == melody[i-1]:
            run += 1
        else:
            if run > 2:
                repeats += (run - 2)
            run = 1
    if run > 2:
        repeats += (run - 2)
    score -= 0.6 * repeats

    # Mild reward for variety
    score += 0.2 * len(set(melody))

    return score

# ----------------------------
# GA operators
# ----------------------------
def tournament_selection(pop: List[Tuple[List[str], float]], k=TOURNAMENT_K) -> List[str]:
    chosen = random.sample(pop, k)
    chosen.sort(key=lambda x: x[1], reverse=True)
    return chosen[0][0].copy()

def one_point_crossover(a: List[str], b: List[str]) -> Tuple[List[str], List[str]]:
    L = len(a)
    if L < 2:
        return a.copy(), b.copy()
    pt = random.randint(1, L-1)
    return a[:pt] + b[pt:], b[:pt] + a[pt:]

def mutate(melody: List[str], rate=MUTATION_RATE) -> List[str]:
    m = melody.copy()
    L = len(m)
    for i in range(L):
        if random.random() < rate:
            op = random.random()
            if op < 0.6:
                m[i] = random.choice(ALLOWED_NOTES)
            elif op < 0.85:
                idx = NOTE_NAMES.index(m[i])
                delta = random.choice([-1,1])
                new_idx = min(max(idx + delta, 0), len(NOTE_NAMES)-1)
                m[i] = NOTE_NAMES[new_idx]
            else:
                j = random.randrange(L)
                m[i], m[j] = m[j], m[i]
    return m

# ----------------------------
# GA main loop
# ----------------------------
def run_ga(pop_size=POP_SIZE, gens=GENERATIONS, melody_len=MELODY_LEN, seed=SEED, verbose=True):
    random.seed(seed)
    np.random.seed(seed)
    pop = []
    for _ in range(pop_size):
        mel = random_melody(melody_len)
        pop.append((mel, fitness(mel)))

    history_best = []
    for g in range(gens):
        pop.sort(key=lambda x: x[1], reverse=True)
        best = pop[0]
        history_best.append(best[1])
        if verbose and (g % max(1, gens//10) == 0):
            print(f"[GA] Generation {g}/{gens}  best fitness {best[1]:.3f}")

        new_pop = pop[:ELITISM]  # elitism preserve top
        while len(new_pop) < pop_size:
            a = tournament_selection(pop)
            b = tournament_selection(pop)
            if random.random() < CROSSOVER_RATE:
                c1, c2 = one_point_crossover(a, b)
            else:
                c1, c2 = a.copy(), b.copy()
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.append((c1, fitness(c1)))
            if len(new_pop) < pop_size:
                new_pop.append((c2, fitness(c2)))
        pop = new_pop

    pop.sort(key=lambda x: x[1], reverse=True)
    return pop, history_best

# ----------------------------
# Output helpers: text, MIDI, optional WAV via fluidsynth
# ----------------------------
def save_melody_text(melody: List[str], fitness_score: float, seed: int, output_dir=OUTPUT_DIR):
    fname = os.path.join(output_dir, f"bhairav_melody_seed{seed}.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"# Bhairav melody generated with seed {seed}\n")
        f.write(" ".join(melody) + "\n")
        f.write(f"# Fitness: {fitness_score}\n")
    return fname

def save_midi(melody: List[str], seed: int, tempo_bpm=TEMPO_BPM, output_dir=OUTPUT_DIR):
    if mido is None:
        raise RuntimeError("mido not installed. Please install with: pip install mido")
    midi_notes = melody_to_midi_notes(melody)
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=bpm2tempo(tempo_bpm)))
    ticks = mid.ticks_per_beat
    for n in midi_notes:
        track.append(Message('note_on', note=int(n), velocity=80, time=0))
        track.append(Message('note_off', note=int(n), velocity=64, time=ticks))
    outpath = os.path.join(output_dir, f"bhairav_melody_seed{seed}.mid")
    mid.save(outpath)
    return outpath

def render_wav_with_fluidsynth(midi_path: str, wav_path: str, soundfont_path: str):
    """Try to render MIDI to WAV using fluidsynth command-line (must be installed)."""
    fluidsynth_exe = shutil.which("fluidsynth") or shutil.which("fluidsynth.exe")
    if fluidsynth_exe is None:
        return False, "fluidsynth not found on PATH"
    if not os.path.isfile(soundfont_path):
        return False, f"SoundFont not found: {soundfont_path}"
    cmd = [fluidsynth_exe, "-ni", soundfont_path, midi_path, "-F", wav_path, "-r", "44100"]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, "OK"
    except subprocess.CalledProcessError as e:
        return False, f"fluidsynth failed: {e}"

# ----------------------------
# CLI / main
# ----------------------------
def main_cli():
    parser = argparse.ArgumentParser(description="Generate Raag Bhairav melodies with GA and save MIDI.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pop", type=int, default=POP_SIZE)
    parser.add_argument("--gens", type=int, default=GENERATIONS)
    parser.add_argument("--len", type=int, default=MELODY_LEN)
    parser.add_argument("--out", type=str, default=OUTPUT_DIR)
    parser.add_argument("--render-wav", action="store_true", help="Attempt to render WAV using FluidSynth and SOUNDFONT_PATH")
    parser.add_argument("--soundfont", type=str, default=SOUNDFONT_PATH, help="Path to .sf2 soundfont for FluidSynth")
    args = parser.parse_args()

    outdir = args.out
    os.makedirs(outdir, exist_ok=True)

    print(f"Running GA: pop={args.pop}, gens={args.gens}, len={args.len}, seed={args.seed}")
    pop, history = run_ga(pop_size=args.pop, gens=args.gens, melody_len=args.len, seed=args.seed, verbose=True)

    best_melody, best_score = pop[0]
    print("\nBest melody (notation):", " ".join(best_melody))
    print("Best fitness:", best_score)

    # save text
    textpath = save_melody_text(best_melody, best_score, args.seed, output_dir=outdir)
    print("Saved melody text to", textpath)

    # save fitness plot
    try:
        plt.figure(figsize=(6,3))
        plt.plot(history)
        plt.title("GA best fitness per generation")
        plt.xlabel("Generation")
        plt.ylabel("Best fitness")
        pngp = os.path.join(outdir, f"fitness_history_seed{args.seed}.png")
        plt.tight_layout()
        plt.savefig(pngp)
        plt.close()
        print("Saved fitness history plot to", pngp)
    except Exception as e:
        print("Could not save fitness plot:", e)

    # write MIDI using mido (pure-Python)
    if mido is None:
        print("mido not installed. Install with: pip install mido to save MIDI files.")
    else:
        midipath = save_midi(best_melody, args.seed, tempo_bpm=TEMPO_BPM, output_dir=outdir)
        print("Saved MIDI to", midipath)
        # Optionally render WAV
        if args.render_wav:
            sf = args.soundfont
            if not sf:
                print("No soundfont provided; cannot render WAV. Provide --soundfont / set SOUNDFONT_PATH.")
            else:
                wavpath = os.path.join(outdir, f"bhairav_melody_seed{args.seed}.wav")
                ok, msg = render_wav_with_fluidsynth(midipath, wavpath, sf)
                if ok:
                    print("Rendered WAV to", wavpath)
                else:
                    print("Failed to render WAV:", msg)

    # print top 5 melodies
    print("\nTop 5 melodies (notation and fitness):")
    for i,(mel,sc) in enumerate(pop[:5]):
        print(f"{i+1:02d}. {' '.join(mel)}  (fitness={sc:.3f})")

    print("\nDone. Output folder:", outdir)
    print("If you want WAV files, install fluidsynth and provide a SoundFont, then run with --render-wav --soundfont /path/to/sf2")

if __name__ == "__main__":
    main_cli()
