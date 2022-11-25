# Sketching the Expression: Flexible Rendering of Expressive Piano Performance with Self-Supervised Learning
* Paper: https://arxiv.org/abs/2208.14867
* Demo: https://free-pig-6c6.notion.site/DEMO-c20a1fea7a0844468a05b971c3b9ef3c


## Installation

Clone this repository, and install the required packages: 

```
git clone https://github.com/rsy1026/sketching_piano_expression.git
```

```
pip install -r requirements.txt
```

Then run this command to make "sketching_piano_expression" a python package for importing inner functions easily:

```
pip install -e .
```

## Parsing features for training
We should align three files to parse the features: score MusicXML, score MIDI and human performance MIDI. Raw data samples can be found in './scripts/data/data_samples/raw_samples'.

```
python3 extract_features.py --xml [filename].musicxml --score [filename].mid --perform [filename].mid --measures [num1] [num2]
```

Please make sure that MusicXML data has no error: 
* Please check if any notes are *hidden* by some annotations (such as trills, glissandos, etc.) in the MusicXML score (You may check with the *MuseScore* software).
* Trills should not be abbreviated as "tr." signs but all notes should appear within each trill.
* Make sure that any complex techniques written in the MusicXML score are not abbreviated as annotations but every single notes should be written.
* Functions to extract the features may be updated in the future for imperfectly aligned MusicXML-MIDI files. These functions may flexibly skip the unaligned notes.