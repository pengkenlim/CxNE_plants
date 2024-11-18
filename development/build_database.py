# %%
import pickle
import os
import sys
def generate_gene_nodes(gene_dict, taxid, sci_name, outpath):
    assert type(taxid) == int
    assert type(sci_name) == str
    with open(outpath, "w") as fout:
        for gene, gid in gene_dict.items():
            fout.write(f"{gene},{taxid},{sci_name},Gene\n")

def generate_coexp_rel(gene_dict, adj_mat_zscore, outpath):
    with open(outpath, "w") as fout:
        gene_list = list(gene_dict.keys())
        for source in gene_list:
            source_id = gene_dict[source]
            for target in gene_list[source_id +1:]:
                target_id = gene_dict[target]
                start_end_str = ",".join(sorted([source,target]))
                strength = adj_mat_zscore[source_id,target_id]
                fout.write(f"{start_end_str},{strength:.6f},CO-EXP\n")
            if source_id % 100 ==0:
                print(source_id, "completed")
# %%
indir = "/mnt/md2/ken/CxNE_plants_data/species_data/"
outdir = "/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/angiosperm_human_yeast"
#spe = "taxid29760" 
#sci_name = "Arabidopsis lyrata "
#taxid = 29760
spe_list = ["taxid29760","taxid3694", "taxid3880", "taxid39947", "taxid4577"] 
taxid_list = [29760, 3694, 3880, 39947, 4577]

for taxid, spe in zip(taxid_list,spe_list ):
    gene_dict_path = os.path.join(indir,spe ,"gene_dict.pkl" )
    with open(gene_dict_path, "rb") as fin:
        gene_dict = pickle.load(fin)


    if True:
        adj_mat_zscore_path = os.path.join(indir,spe, "adj_mat_zscore.pkl")
        with open(adj_mat_zscore_path, "rb") as fin:
            adj_mat_zscore = pickle.load( fin)

    if False:
        pass
        #gene_nodes_path  =  os.path.join(outdir, f"{spe}_Gene_nodes.csv")
        #generate_gene_nodes(gene_dict, taxid, sci_name, gene_nodes_path)

    if True:
        coexp_rel_path =  os.path.join(outdir, f"{spe}_CO-EXP_relationships.csv")
        generate_coexp_rel(gene_dict, adj_mat_zscore, coexp_rel_path)
# %%
#mine plantcyc
import os
import sys


import requests
from collections import Counter

def get_pathways(PMNCODE):
    met_annot_dict = {}
    exp_pathways_URL=f"https://pmn.plantcyc.org/{PMNCODE}/search-query?type=PATHWAY"
    page_string = requests.get(exp_pathways_URL).text
    
    pathway_codes = page_string.split(f"/pathway?orgid={PMNCODE}&id=")[1:]
    pathway_codes = [chunk.split("</A")[0] for chunk in pathway_codes]
    for PWY_NAME in pathway_codes:
        met_annot_dict[PWY_NAME.split("\">")[0]] = {"NAME" : PWY_NAME.split("\">")[1]}

    return met_annot_dict


def mine_genes_from_pathway(PMNCODE, met_annot_dict):
    for P_idx, PWY in enumerate(met_annot_dict.keys(), start = 1):
        URL=f"https://pmn.plantcyc.org/{PMNCODE}/pathway-genes?object=" + PWY
        page_string = requests.get(URL).text.split("\n")
        NAME= page_string[0]
        Genes={}
        for line in page_string[3:]:
            if line != "":
                geneID = line.split("\t")[1].upper()
                Evidence  = line.split("\t")[-1]
                Genes[geneID]= {"RXN": line.split("\t")[3], "EVD": Evidence}
        met_annot_dict[PWY]["Genes"] = Genes
        met_annot_dict[PWY]["Name"] = NAME

        if P_idx % 10 == 0:
            print(P_idx, "pathways mined")
    return met_annot_dict

def edge_dump_deprecated(met_annot_dict, path ,type="Cri_1"):
    string = []
    for PWY , info in met_annot_dict.items():
        Name = info["Name"]
        for edge in info["Edges"][type]:
            string.append(f"{edge}\t{PWY}\t{Name}\n")
    string = "".join(string)
    with open(path, "w") as f:
        f.write(string)

def edge_dump(met_annot_dict, path ,type="Cri_1"):
    edge_dict = {}
    for PWY , info in met_annot_dict.items():
        Name = info["Name"]
        for edge in info["Edges"][type]:
            if edge not in edge_dict.keys():
                edge_dict[edge]={"PWY":PWY, "Name":Name}
            else:
                edge_dict[edge]={"PWY":edge_dict[edge]["PWY"] + "|" + PWY, "Name":edge_dict[edge]["Name"]+ "|" + Name} #for when an edge is belongs to more than one pathway
    string = []
    for edge, info in edge_dict.items():
        Name = info["Name"]
        PWY = info["PWY"]
        string.append(f"{edge}\t{PWY}\t{Name}\n")
    string = "".join(string)
    with open(path, "w") as f:
        f.write(string)

def extract_edges(met_annot_dict,type="Cri_1"):
    positive_met_edges = []
    for PWY , info in met_annot_dict.items():
        for edge in info["Edges"][type]:
            positive_met_edges.append(edge)
    return list(set(positive_met_edges)) #to get rid of duplicates


# %%
species2PMNCODE_dict = {"taxid3702": "ARA",
                        "taxid29760": "GRAPE",
                        "taxid3694": "POPLAR",
                        "taxid3711": "CHINESECABBAGE",
                        "taxid3880": "MTRUNCATULA",
                        "taxid39947": "ORYZA",
                        "taxid4081": "TOMATO",
                        "taxid4577": "CORN",
                        "taxid59689": "ALYRATA",
                        "taxid81970":"AHALLERI",}
species_met_annot_dict = {}
for sp, PMNCODE in species2PMNCODE_dict.items():
    met_annot_dict = get_pathways(PMNCODE)
    met_annot_dict = mine_genes_from_pathway(PMNCODE, met_annot_dict)
    print(PMNCODE)
    print(len(met_annot_dict))
    species_met_annot_dict[sp] = met_annot_dict
# %%
import pickle
species_data_dir = "/mnt/md2/ken/CxNE_plants_data/species_data/"

master_met_annot_dict = {}
for sp in species2PMNCODE_dict.keys():
    gene_dict_path = os.path.join(species_data_dir ,sp, "gene_dict.pkl")
    with open(gene_dict_path, "rb") as fbin:
        gene_dict = pickle.load(fbin)
    for PWY, PWY_info in species_met_annot_dict[sp].items():
        for GENE, GENE_info in PWY_info["Genes"].items():
            try:
                catch = gene_dict[GENE] # check if geneid is valid
                try:
                    master_met_annot_dict[PWY]["Genes"][GENE] = GENE_info
                except:# means first time pathway was seen
                    master_met_annot_dict[PWY]= {"Genes" : {GENE: GENE_info},
                                                "Name" : PWY_info["Name"]}       
            except:
                pass

# %%
#lets save these precious dictionaries
base_dir = "/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/plantcyc"
species_met_annot_dict_path= os.path.join(base_dir, "species_met_annot_dict.pkl")
master_met_annot_dict_path = os.path.join(base_dir, "master_met_annot_dict.pkl")

with open(species_met_annot_dict_path, "wb") as fbout:
    pickle.dump(species_met_annot_dict,fbout )

with open(master_met_annot_dict_path, "wb") as fbout:
    pickle.dump(master_met_annot_dict,fbout )
# %%
def read_interproscan_file(path_to_annot_file, EVDCODE = "ISS"):
    species_GO_edges = []
    with open(path_to_annot_file, "r") as fin:
        for line in fin:
            if line != "":
                line_split = line.split("\n")[0].split("\t")
                GOTERMS = line_split[-2]
                if GOTERMS != "-":
                    GENEID = line_split[0]
                    SOURCE = line_split[3]
                    REASON_ID = line_split[4]
                    REASON = line_split[5].replace("\"","")
                    GOTERMS = [term.split("(")[0] for term in GOTERMS.split("|")]
                    for term in GOTERMS:
                        species_GO_edges.append({"GENEID": GENEID,
                                                "term": term,
                                                "REASON_ID": REASON_ID,
                                                "REASON": REASON,
                                                "SOURCE": SOURCE,
                                                "EVDCODE" : EVDCODE})
    return species_GO_edges
            
def generate_GO_rel(species_GO_edges, GO_relationship_path, tid2gid_dict):
    rel_set = set()
    for edge_info in species_GO_edges:
            GENEID = edge_info["GENEID"]
            term = edge_info["term"]
            REASON_ID = edge_info["REASON_ID"]
            REASON = edge_info["REASON"]
            SOURCE = edge_info["SOURCE"]
            EVDCODE = edge_info["EVDCODE"]
            try:
                rel_set.add(f"{tid2gid_dict[GENEID]},{term},{REASON_ID},\"{REASON}\",{SOURCE},{EVDCODE},HAS_GO_TERM\n")
            except:
                pass
    rel_set = list(rel_set)
    with open(GO_relationship_path, "w") as fout:
        for rel in rel_set :
            fout.write(rel)



# %%

import os
import pickle

##interproscan
EVDCODE = "ISS"
interproscan_out_dir  = "/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/interproscan/"
file_name = "taxid4577_interproscan.tsv"
spe = "taxid4577"
output_dir = "/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/angiosperm_human_yeast"

path_to_annot_file = os.path.join(interproscan_out_dir, file_name)

species_GO_edges =read_interproscan_file(path_to_annot_file, EVDCODE = EVDCODE)
GO_relationship_path = os.path.join(output_dir, f"{spe}_HAS_GO_TERM_relationships.csv")

if True:
    path_to_annot_file_2 = os.path.join(interproscan_out_dir, file_name.split(".tsv")[0]+ "_expanded.tsv")
    species_GO_edges_2 =read_interproscan_file(path_to_annot_file_2, EVDCODE = EVDCODE)

    species_GO_edges += species_GO_edges_2

#load tid2gid
species_data_dir = "/mnt/md2/ken/CxNE_plants_data/species_data/"
Tid2Gid_dict_path = os.path.join(species_data_dir, spe, "Tid2Gid_dict.pkl")
with open(Tid2Gid_dict_path, "rb") as fbin:
    Tid2Gid_dict = pickle.load(fbin)




generate_GO_rel(species_GO_edges, GO_relationship_path, Tid2Gid_dict)


# %%
# %%
import pickle

#lets LOAD these precious dictionaries
base_dir = "/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/plantcyc"
species_met_annot_dict_path= os.path.join(base_dir, "species_met_annot_dict.pkl")
master_met_annot_dict_path = os.path.join(base_dir, "master_met_annot_dict.pkl")

with open(species_met_annot_dict_path, "rb") as fbin:
    species_met_annot_dict= pickle.load(fbin )

with open(master_met_annot_dict_path, "rb") as fbin:
    master_met_annot_dict= pickle.load(fbin )
# %%
#generate rel_nodes
output_dir = "/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/angiosperm_human_yeast"

PWY_nodes = set()
for PWY, PWY_info in master_met_annot_dict.items():
    Name = PWY_info["Name"]
    PWY_nodes.add(f"{PWY},\"{Name}\",BioCycPathway\n")

node_path = os.path.join(output_dir , "BioCycPathway_nodes.csv")
with open(node_path, "w") as fout:
    for node in PWY_nodes:
        fout.write(node)
# %%
#generate metabolic edges
output_dir = "/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/angiosperm_human_yeast"
species_data_dir = "/mnt/md2/ken/CxNE_plants_data/species_data/"
for spe, met_annot_dict in species_met_annot_dict.items():
    rel_path = os.path.join(output_dir, f"{spe}_IN_BIOCYC_PATHWAY_relationships.csv")
    rel_set = set()
    gene_dict_path = os.path.join(species_data_dir ,spe, "gene_dict.pkl")
    with open(gene_dict_path, "rb") as fbin:
        gene_dict = pickle.load(fbin)
    for PWY, PWY_info in met_annot_dict.items():
       for Gene, Gene_info in PWY_info["Genes"].items():
            try:
                catch = gene_dict[Gene]
                RXN , EVD = Gene_info["RXN"], Gene_info["EVD"]
                edge = ",".join([ Gene, PWY, "\""+RXN+ "\"", EVD, "IN_BIOCYC_PATHWAY"]) + "\n"
                rel_set.add(edge)
            except:
                pass
    rel_set = list(rel_set)
    with open(rel_path, "w") as fout:
        for edge in rel_set:
            fout.write(edge)
            
# %%
# pathway_classes
pathway_parents_path = "/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/plantcyc/PWY_pathway_parents_PWY_merged.tsv"

PWY2parents_dict = {}
with open(pathway_parents_path, "r") as fin:
    for line_no , line in enumerate(fin):
        if line_no != 0  and line != "":
            line_contents = line.split("\n")[0].split("\t")
            PWY = line_contents[0]
            parents = line_contents[1].split(" // ")
            parents_id = line_contents[2].split(" // ")
            
            PWY2parents_dict[PWY] = {}
            for parent_id, parent in zip(parents_id, parents):
                PWY2parents_dict[PWY][parent_id] = parent



# %%
#generate pathway class nodes.csv and rel
PWT_class_nodes = set()
PWT_class_rel = set()
for PWY, PWY_info in master_met_annot_dict.items():
    try:
        for parent_id, parent in PWY2parents_dict[PWY].items():
            PWT_class_nodes.add(f"{parent_id},\"{parent}\",BioCycPathwayClass\n")
            PWT_class_rel.add(f"{PWY},{parent_id},IN_BIOCYC_PATHWAY_CLASS\n")
    except:
        pass

PWT_class_nodes = list(PWT_class_nodes)
PWT_class_rel = list(PWT_class_rel)
PWT_class_nodes_path = os.path.join(output_dir, "BioCycPathwayClass_nodes.csv")
PWT_class_rel_path = os.path.join(output_dir, "IN_BIOCYC_PATHWAY_CLASS_relationships.csv")

with open(PWT_class_nodes_path, "w") as fout:
    for node in PWT_class_nodes:
        fout.write(node)

with open(PWT_class_rel_path, "w") as fout:
    for rel in PWT_class_rel:
        fout.write(rel)


# %%
ATH_GO_SLIM_PATH = "/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/GO_downloads/ATH_GO_GOSLIM.txt"
gene_dict_path = "/mnt/md2/ken/CxNE_plants_data/species_data/taxid3702/gene_dict.pkl"
with open(gene_dict_path, "rb") as fbin:
    gene_dict = pickle.load(fbin)

# %%
HAS_GO_TERM_rel = set()
with open(ATH_GO_SLIM_PATH, "r") as fin:
    for line in fin:
        if "!" not in line and line !="":
            line_contents = line.split("\n")[0].split("\t")
            try:
                GENE = line_contents[0]
                catch = gene_dict[GENE] # search if geneid is valid
                REASON = line_contents[-5].replace("\"","")
                REASON_ID = line_contents[-3].replace(" ","")
                SOURCE = "TAIR"
                EVD = line_contents[-6]
                GO_TERM = line_contents[5]
                HAS_GO_TERM_rel.add(f"{GENE},{GO_TERM},{REASON_ID},\"{REASON}\",{SOURCE},{EVD},HAS_GO_TERM\n")
            except:
                pass
# %%
HAS_GO_TERM_rel_path = os.path.join( output_dir , "taxid3702_HAS_GO_TERM_relationships.csv")

with open(HAS_GO_TERM_rel_path, "a") as fout:
    for edge in HAS_GO_TERM_rel:
        fout.write(edge)

# %%
#scan the relationships to make the modes
GO_nodes = {}
scan_dir = outdir
for file in os.listdir(scan_dir):
    if "HAS_GO_TERM_relationships" in file:
        path = os.path.join( scan_dir, file)
        with open(path, "r") as fin:
            for line in fin:
                GO_nodes[line.split(",")[1]] = ""
# %%
#read OBO
OBO_file_path = "/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/GO_downloads/go-basic.obo"
with open(OBO_file_path, "r") as fin:
    contents = fin.read()
contents_chunk = contents.split("[Term]\nid: ")[1:]

contents_chunk = {chunk.split("\nname")[0]: {"name":chunk.split("\nname: ")[1].split("\n")[0],  "type":chunk.split("\nnamespace: ")[1].split("\n")[0] , "description": chunk.split("\ndef: \"")[1].split("\"")[0]} for chunk in contents_chunk}

# %%
GO_nodes_set = set()
for GO_node , _ in GO_nodes.items():
    info = contents_chunk[GO_node]
    NAME = info["name"]
    TYPE = info["type"]
    DESCRIPTION = info["description"]
    GO_nodes_set.add(f"{GO_node},\"{NAME}\",{TYPE},\"{DESCRIPTION}\",GOTerm\n")

GO_nodes_path = os.path.join( outdir, "GOTerm_nodes.csv")
GO_nodes_set = list(GO_nodes_set)

with open(GO_nodes_path, "w") as fout:
    for node in GO_nodes_set:
        fout.write(node)

# %%
def read_interproscan_file(path_to_annot_file, EVDCODE = "ISS"):
    species_GO_edges = []
    with open(path_to_annot_file, "r") as fin:
        for line in fin:
            if line != "":
                line_split = line.split("\n")[0].split("\t")
                GOTERMS = line_split[-2]
                if GOTERMS != "-":
                    GENEID = line_split[0]
                    SOURCE = line_split[3]
                    REASON_ID = line_split[4]
                    REASON = line_split[5].replace("\"","")
                    GOTERMS = [term.split("(")[0] for term in GOTERMS.split("|")]
                    for term in GOTERMS:
                        species_GO_edges.append({"GENEID": GENEID,
                                                "term": term,
                                                "REASON_ID": REASON_ID,
                                                "REASON": REASON,
                                                "SOURCE": SOURCE,
                                                "EVDCODE" : EVDCODE})
    return species_GO_edges

# %%
#generate description
import pickle
outdir = "/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/angiosperm_human_yeast/"
sci_name = "Arabidopsis halleri"
taxid = 81970
spe = f"taxid{taxid}"
spe_data_dir = "/mnt/md2/ken/CxNE_plants_data/species_data/"
fasta_dir = "/mnt/md0/ken/correlation_networks/Plant-GCN_data/Orthofinder/ATTED_input_fasta/"


#load gene dict
gene_dict_path = os.path.join(spe_data_dir, spe, "gene_dict.pkl")
Tid2Gid_dict_path = os.path.join(spe_data_dir, spe, "Tid2Gid_dict.pkl")
fasta_path = os.path.join( fasta_dir, spe+ ".fasta")

with open(gene_dict_path, "rb") as fbin:
    gene_dict = pickle.load(fbin)

with open(Tid2Gid_dict_path, "rb") as fbin:
    Tid2Gid_dict = pickle.load(fbin)

with open(fasta_path, "r") as fin:
    contents = fin.read()

fasta_contents = {Tid2Gid_dict.get(chunk.split("\n")[0].split(" ")[0],"NIL") : "".join(chunk.split("\n")[1:]) for chunk in contents.split(">")[1:]}


#description from interproscan


path_to_annot_file =os.path.join("/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/interproscan", spe+ "_interproscan_full.tsv")
species_GO_edges = read_interproscan_file(path_to_annot_file)

genedesc_dict = {}
#for arabidopsis only
if False:
    tair_desc_file_path = os.path.join("/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/GO_downloads", "tair_gene_des.txt")

    with open(tair_desc_file_path, "r") as fin:
        for line_no , line in enumerate(fin):
            if line_no > 1 and line != "":
                line_contents = line.split("\n")[0].split("\t")
                geneid = line_contents[0]
                desc = line_contents[2]
                if "source" not in desc:
                    desc = desc + ";(source:TAIR)."
                try:
                    catch = gene_dict[geneid]
                    try:
                        genedesc_dict[geneid][desc] = ""
                    except:
                        genedesc_dict[geneid] = {}
                        genedesc_dict[geneid][desc] = ""
                except:
                    pass

#genedesc
SOURCE_targets = [ "PANTHER","NCBIfam", "SFLD" , "Pfam"]


for SOURCE_target in SOURCE_targets:
    for edge in species_GO_edges:
        GENEID_t = edge["GENEID"]
        SOURCE = edge["SOURCE"]
        if SOURCE == SOURCE_target:
            try:
                GENEID = Tid2Gid_dict[GENEID_t]
                catch = gene_dict[GENEID]
                REASON = edge["REASON"]
                try:
                    genedesc_dict[GENEID][REASON + f";(source:{SOURCE})."] = ""
                except:
                    genedesc_dict[GENEID] = {}
                    genedesc_dict[GENEID][REASON + f";(source:{SOURCE})."] = ""
            except:
                pass



gene_node_path = os.path.join(outdir,f"{spe}_updated_gene_nodes.csv")
gene_nodes_set = set()
with open(gene_node_path, "w") as fout:
    for gene, _ in gene_dict.items():
        cdna = fasta_contents.get(gene, "-")
        desc = genedesc_dict.get(gene ,"Unknown protein without domain matches.")
        if desc!= "Unknown protein without domain matches.":
            desc = list(desc.keys())
            desc = " | ".join(desc)
            desc = desc.replace("\"", "")
        gene_nodes_set.add(f"{gene},{taxid},{sci_name},\"{desc}\",{cdna},Gene\n")

with open(gene_node_path, "w") as fout:
    for node in list(gene_nodes_set):
        fout.write(node)


# %%
