# AVS - Analog Vinyl System
This application is specifically crafted to assist novice DJs in their journey of mixing by utilizing authentic vinyl 
records, similar to DVS (Digital Vinyl System). Instead of employing timecode vinyls, it functions by streaming audio 
from multiple vinyl decks using genuine vinyl records. The program showcases the waveform and BPM (beats per minute), 
providing users with visual guidance throughout the process. 

To capture the output from each deck, an audio interface or USB mixer is necessary.

# Features:
Display waveform for each deck.
Display BPM for each deck.

# WIP:
Pyaudio crashes from buffer overflow + wierd artifacts are present.

# TODO:
rewrite the system that spawns waveforms for each deck with a for loop
Pyaudio crashes from buffer overflow + wierd artifacts are present.
Add config GUI + file with names of audio interface inputs
BPM Possibly could be improved by using LPF.
Display could be improved by colouring the waveform using spectrum.
Disable autoscaling, add setting for sensitivity
add zoom levels.
Make GUI to look like friendly DJ Software
Display key