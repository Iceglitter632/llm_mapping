import torch
import torch.nn.functional as F

from miditoolkit import MidiFile
from miditok import TokenizerConfig, REMI

from .base import InputEncoder

class MidiEncoder(InputEncoder):
    def __init__(self, args):
        PITCH_RANGE = (21, 109)
        BEAT_RES = {(0, 1): 8, (1, 2): 4, (2, 4): 2, (4, 8): 1}
        NUM_VELOCITIES = 24
        SPECIAL_TOKENS = ["PAD", "MASK", "BOS", "EOS"]
        USE_CHORDS = True
        USE_RESTS = False
        USE_TEMPOS = True
        USE_TIME_SIGNATURE = False
        USE_PROGRAMS = True
        NUM_TEMPOS = 32
        TEMPO_RANGE = (50, 200)
        TOKENIZER_PARAMS = {
            "pitch_range": PITCH_RANGE,
            "beat_res": BEAT_RES,
            "num_velocities": NUM_VELOCITIES,
            "special_tokens": SPECIAL_TOKENS,
            "use_chords": USE_CHORDS,
            "use_rests": USE_RESTS,
            "use_tempos": USE_TEMPOS,
            "use_time_signatures": USE_TIME_SIGNATURE,
            "use_programs": USE_PROGRAMS,
            "num_tempos": NUM_TEMPOS,
            "tempo_range": TEMPO_RANGE,
        }
        self.config = TokenizerConfig(**TOKENIZER_PARAMS)
        self.encoder = REMI(self.config)
        self.vocab_len = len(self.encode.vocab)
    
    def get_ground_truth(self, midi):
        return midi["input_ids"]
    
    def get_onehot(self, ground_truth_tokens):
        return F.one_hot(ground_truth_tokens, num_classes=self.vocab_len).float().to(self.device)
