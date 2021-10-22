import numpy as np

def read_dat(file_path):
    lines = list()
    for line in open(file_path, 'r'):
        line = line.replace('\n', '')
        line = line.split('\t')
        lines.append(line)

    return lines
        
