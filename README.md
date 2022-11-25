# Sketching the Expression: Flexible Rendering of Expressive Piano Performance with Self-Supervised Learning
* Paper: https://arxiv.org/abs/2208.14867
* Demo: https://free-pig-6c6.notion.site/DEMO-c20a1fea7a0844468a05b971c3b9ef3c


## Installation

Clone this repository, and install the required packages: 

'''
git clone https://github.com/rsy1026/sketching_piano_expression.git
pip install -r requirements.txt
'''

Then run this command to make "sketching_piano_expression" a python package for importing inner functions easily:

```
pip install -e .
```

## Parsing features for training
We should align three files to parse the features: score MusicXML, score MIDI and human performance MIDI. Raw data samples can be found in './scripts/data/data_samples/raw_samples'.