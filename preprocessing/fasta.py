# %%
#import 
#from Bio.Seq import Seq

#def read_fasta(path):
#    with open(path, "r") as fin:
#        contents = fin.read().split(">")[1:]
#    return contents
# %%
#fasta = read_fasta("/mnt/md0/ken/correlation_networks/Plant-GCN_data/Orthofinder/ATTED_input_fasta/taxid3702.fasta")
#fasta = {line.split("\n")[0].split(" |")[0]: "".join(line.split("\n")[1:]) for line in fasta}
# %%
#seq = Seq(fasta["AT1G08710.2"])

# %%
#help(seq.startswith)
# %%
#output = seq.translate(table='Standard', to_stop=False)
# %%
#dir(seq)
# %%
import os
import sys

orfFinder_path = "/home/ken/ORFfinder"

fasta_dir = "/mnt/md0/ken/correlation_networks/Plant-GCN_data/Orthofinder/ATTED_input_fasta/"
outputdir = "/mnt/md0/ken/correlation_networks/Plant-GCN_data/ORFfinder"
for file in os.listdir(fasta_dir):
    if "fasta" in file:
        input_file = os.path.join(fasta_dir,file)
        output_file = os.path.join(outputdir, file.split(".fasta")[0] + ".pep")
        command = f"{orfFinder_path} -outfmt 0  -s 1  -ml 10 -n true -strand plus -in {input_file} -out {output_file}"
        print(f"running\n{command}")
        os.system(command)


# %%
outputdir = "/mnt/md0/ken/correlation_networks/Plant-GCN_data/ORFfinder"
taxid = "taxid81970"
#read fasta
pep_path  = os.path.join(outputdir, taxid + ".pep")
with open(pep_path, "r") as fin:
    contents = fin.read().split(">")

transcript_2_ORF_dict = {}
for line in contents:
    if line != "":
        tid = "_".join(line.split("\n")[0].split("_")[1:]).split(":")[0]
        length = len("".join(line.split("\n")[1:]))
        if "partial" in line:
            type = "partial"
        else:
            type  = "full"
        try:
            transcript_2_ORF_dict[tid]["INFO"].append(line)
            transcript_2_ORF_dict[tid]["LENGTH"].append(length)
            transcript_2_ORF_dict[tid]["TYPE"].append(type)
        except:
            transcript_2_ORF_dict[tid] = {}
            transcript_2_ORF_dict[tid]["INFO"] = [line]
            transcript_2_ORF_dict[tid]["LENGTH"] = [length]
            transcript_2_ORF_dict[tid]["TYPE"] = [type]


import numpy as np

transcript_2_ORF_dict_filtered = {}
for tid, tid_dict in transcript_2_ORF_dict.items():
    types = np.array(tid_dict["TYPE"]) == "full"
    if True in types:
        max_length = max( np.array(tid_dict["LENGTH"])[types] )
        max_length_idx = tid_dict["LENGTH"].index(max_length)
        info = tid_dict["INFO"][max_length_idx]
    else:
        max_length = max( tid_dict["LENGTH"])
        max_length_idx = tid_dict["LENGTH"].index(max_length)
        info = tid_dict["INFO"][max_length_idx]
    transcript_2_ORF_dict_filtered[tid] = tid + "    "+ info

newfasta = ">" +  ">".join(list(transcript_2_ORF_dict_filtered.values()))

new_pep_path  = os.path.join(outputdir, taxid + "_longest_ORF.pep")
with open(new_pep_path, "w") as fout:
    fout.write(newfasta)
# %%
