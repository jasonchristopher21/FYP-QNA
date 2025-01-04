from typing import List
import random
from datasets import Dataset, concatenate_datasets
import torch 

def get_data_from_mixing_ratio(data_sources: List[Dataset], scores: List[float]) -> Dataset:
    resulting_dataset = sample_from(data_sources, scores, seed=2024)
    return resulting_dataset

# Random sample from a Huggingface Dataset based on a ratio
def random_sample_from_dataset(dataset: Dataset, ratio: float, contaminate=False, seed=None) -> Dataset:
    if seed is None:
        torch.manual_seed()
    else:
        torch.manual_seed(seed)
    num_samples = len(dataset)
    # if contaminate:
    #     num_samples = int(num_samples*0.9)
    indices = random.sample(range(num_samples), k=int(ratio*num_samples))
    sampled_dataset = dataset.select(indices)
    return sampled_dataset

def sample_from(data_sources: List[Dataset], mixing_ratio: List[float], method="random_sample", seed=None):
    if seed is None:
        torch.manual_seed()
    else:
        torch.manual_seed(seed)
    assert len(data_sources) == len(mixing_ratio)
    sampled_data = []
    
    # "random_sample", # randomly sample
    # "highest_influence", # take top K influence
    # "lowest_influence", # take bottom K influence 
    # "influence_sample", # sample based on influence distribution
    # "reverse_influence_sample", # sample based on reverse influence distribution
    # "remove_harmful_then_uniform", # remove bottom 10% influence, then sample from the rest uniformly
    # "remove_harmful_then_follow_IF", # remove bottom 10% influence, then sample from the rest based on influence distribution
    # "remove_tail_ends_then_uniform",
    
    for idx, loader in enumerate(data_sources):
        ratio = mixing_ratio[idx]
        if ratio == 0:
            continue
        if method == "random_sample":
            sampled_data.append(random_sample_from_dataset(loader, ratio, seed=seed))
        # if method == "highest_influence":
        #     sampled_data.append(take_highest_influence(loader, num_datapoints=int(ratio*base_number_of_batches)*batch_size, influences=additional_info[idx], seed=seed))
        # if method == "influence_sample":
        #     sampled_data.append(sample_from_influence(loader, num_datapoints=int(ratio*base_number_of_batches)*batch_size, influences=additional_info[idx], seed=seed))
        # if method == "remove_harmful_then_uniform":
        #     sampled_data.append(remove_lowest_influence_then_sample_uniformly(loader, num_datapoints=int(ratio*base_number_of_batches)*batch_size, influences=additional_info[idx], seed=seed))
        # if method == "remove_harmful_then_follow_IF":
        #     sampled_data.append(remove_lowest_influence_then_sample_based_on_IF(loader, num_datapoints=int(ratio*base_number_of_batches)*batch_size, influences=additional_info[idx], seed=seed))
        # if method == "lowest_influence": # just multiply influence by -1
        #     sampled_data.append(take_highest_influence(loader, num_datapoints=int(ratio*base_number_of_batches)*batch_size, influences=-additional_info[idx], seed=seed))
        # if method == "reverse_influence_sample":  # just multiply by -1
        #     sampled_data.append(sample_from_influence(loader, num_datapoints=int(ratio*base_number_of_batches)*batch_size, influences=-additional_info[idx], seed=seed))
        # if method == "remove_tail_ends_then_uniform":
        #     sampled_data.append(remove_tail_ends_then_uniform(loader, num_datapoints=int(ratio*base_number_of_batches)*batch_size, influences=additional_info[idx], seed=seed))
        
    combined_dataset = concatenate_datasets(sampled_data)
    return combined_dataset

"""
# give agent some instructions/score req/something else and ask them to give dataset
def get_data_from_mixing_ratioOLD(data_sources : List[DataLoader], scores : List[float], additional_info=None, base_number_of_batches=10) -> DataLoader:
    resulting_dataloader = sample_from(data_sources, scores, seed=2024, base_number_of_batches=base_number_of_batches)
    return resulting_dataloader

# def get_data_from_influence_function(data_sources : List[DataLoader], scores : List[float], base_number_of_batches=10) -> DataLoader:
#     resulting_dataloader = None
    
#     return resulting_dataloader

def mixup(agent_data : List[float], mixing_parameter : float): # change data format of agent data (should be a list of datasets, not list of float)
    
    # DATA MIXING CODE GOES HERE
    data = 0.0 # data set format
    
    return data

def get_performance(data):
    # FINETUNE AND FIND PERFORMANCE; CODE GOES HERE
    performance = random.random()
    
    return performance
"""