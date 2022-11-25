import subprocess
from glob import glob
import os 
import shutil
import numpy as np
from .parse_utils import extract_midi_notes

def xml_score_match(xml, score):
    nakamura_c = './MusicXMLToMIDIAlign.sh'
    subprocess.call([nakamura_c, xml, score])

def make_corresp(score, perform, align_c):
    subprocess.call([align_c, score, perform])

def copy_alignment_tools(tool_path, save_path):
    align_c = os.path.join(tool_path, 'MIDIToMIDIAlign.sh')
    code_c = os.path.join(tool_path, 'Code')
    programs_c = os.path.join(tool_path, 'Programs')
    align_copy = os.path.join(save_path, 'MIDIToMIDIAlign.sh')
    programs_copy = os.path.join(save_path, 'Programs')
    code_copy = os.path.join(save_path, 'Code')
    # copy files for every group
    if not os.path.exists(align_copy):
        shutil.copy(align_c, align_copy)
    if not os.path.exists(programs_copy):
        shutil.copytree(programs_c, programs_copy)
    if not os.path.exists(code_copy):
        shutil.copytree(code_c, code_copy)
    os.system('chmod 777 -R {}'.format(align_copy)) 
    os.system('chmod 777 -R {}'.format(os.path.join(programs_copy, "*"))) 
    os.system('chmod 777 -R {}'.format(os.path.join(code_copy, "*"))) 
    
def remove_alignment_tools(tool_path):
    align_copy = os.path.join(tool_path, 'MIDIToMIDIAlign.sh')
    programs_copy = os.path.join(tool_path, 'Programs')
    code_copy = os.path.join(tool_path, 'Code')
    # copy files for every group
    assert os.path.exists(align_copy)
    os.remove(align_copy)
    assert os.path.exists(programs_copy)
    shutil.rmtree(programs_copy)
    assert os.path.exists(code_copy)
    shutil.rmtree(code_copy)

def copy_scores_for_perform(perform_file, score_files, xml_files, p_name):
    xml_file = [xml_ for xml_ in xml_files if p_name in xml_][0]
    score_file = [score_ for score_ in score_files if p_name in score_][0]
    parent_path = os.path.dirname(perform_file)

    shutil.copy(xml_file, os.path.join(
        parent_path, "{}.plain.xml".format(p_name)))
    shutil.copy(score_file, os.path.join(
        parent_path, "{}.plain.mid".format(p_name)))
    print("saved xml/score file for {}".format(p_name))

def save_corresp_file(perform, score, tool_path, save_path, remove_cleaned=False):
    # get filenames 
    _perform = '.'.join(os.path.basename(perform).split('.')[:-1])
    _score = '.'.join(os.path.basename(score).split('.')[:-1])
    p_name = os.path.basename(perform).split('.')[0]
    s_name = os.path.basename(score).split('.')[0]
    # assert p_name == s_name # make sure same filenames
    corresp_path = os.path.join(save_path, '{}.cleaned_corresp.txt'.format(_perform))

    if not os.path.exists(corresp_path):
        # copy if cannot access to alignment tools
        copy_alignment_tools(tool_path, save_path)
        perform_savename = os.path.join(save_path, "{}.cleaned.mid".format(_perform))
        score_savename = os.path.join(save_path, "{}.cleaned.mid".format(_score))
        # temporally save cleaned midi
        extract_midi_notes(perform, 
            clean=False, no_pedal=True, save=True, savepath=perform_savename)
        extract_midi_notes(score, 
            clean=True, save=True, savepath=score_savename)
        # save corresp file
        os.chdir(os.path.dirname(perform))
        make_corresp(_score+".cleaned", _perform+".cleaned", './MIDIToMIDIAlign.sh')
        # erase all resulted files but corresp file
        else_txt = glob('./*[!_corresp].txt'.format(_perform))
        for file_ in else_txt:
            os.remove(file_)
        _perform_txt = './{}.cleaned_spr.txt'.format(_perform)
        _score_txt = './{}.cleaned_spr.txt'.format(_score)
        if os.path.exists(_perform_txt):
            os.remove(_perform_txt)
        if os.path.exists(_score_txt):
            os.remove(_score_txt)
        if remove_cleaned is True:
            _perform_clean = './{}.cleaned.mid'.format(_perform)
            _score_clean = './{}.cleaned.mid'.format(_score)
            if os.path.exists(_perform_clean):
                os.remove(_perform_clean)
            if os.path.exists(_score_clean):
                os.remove(_score_clean)
        remove_alignment_tools("./")
        print('saved corresp file for {}'.format(_perform))
        print()
    
    return corresp_path






