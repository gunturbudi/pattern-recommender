from typing import List
import numpy as np
from rich.console import Console
from rich.table import Table

def dcg_at_k(r: List[int], k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r: List[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k."""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def calculate_metrics(recommendations: List[List[str]], ideal: List[List[str]]) -> dict:
    max_k = 5  # Maximum rank for NDCG calculations
    ndcg_scores = {k: [] for k in range(1, max_k + 1)}
    ndcg_details = []

    for rec, ideal_rec in zip(recommendations, ideal):
        # Convert to graded relevance
        relevance = []
        for i in range(len(rec)):
            if rec[i] in ideal_rec:
                pos_diff = abs(ideal_rec.index(rec[i]) - i)
                relevance_score = max(5 - pos_diff, 1)
            else:
                relevance_score = 0
            relevance.append(relevance_score)

        row_ndcg = {}
        for k in range(1, max_k + 1):
            ndcg_score = ndcg_at_k(relevance, k)
            ndcg_scores[k].append(ndcg_score)
            row_ndcg[f'NDCG@{k}'] = ndcg_score
        ndcg_details.append(row_ndcg)

    metrics = {
        "mean_ndcg": {k: np.mean(ndcg_scores[k]) for k in ndcg_scores},
        "ndcg_details": ndcg_details
    }

    return metrics

def find_best_ndcg_rows(ndcg_details):
    best_rows = {}
    for k in range(1, 6):  # For each NDCG rank from 1 to 5
        best_score = -1
        best_row_index = -1
        for i, row in enumerate(ndcg_details):
            if row[f'NDCG@{k}'] > best_score:
                best_score = row[f'NDCG@{k}']
                best_row_index = i
        best_rows[f'best_row_for_ndcg@{k}'] = (best_row_index, best_score)
    return best_rows

def sort_ndcg_rows(ndcg_details):
    sorted_rows = {}
    for k in range(1, 6):  # For each NDCG rank from 1 to 5
        rows_with_scores = [(i, row[f'NDCG@{k}']) for i, row in enumerate(ndcg_details)]
        rows_with_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by NDCG score, highest first
        sorted_rows[f'ndcg@{k}_sorted'] = rows_with_scores
    return sorted_rows

def display_sorted_ndcg_rows(sorted_ndcg_rows):
    console = Console()
    for k in sorted_ndcg_rows:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Row Index", style="dim")
        table.add_column(f"NDCG@{k[-1]}", justify="right")

        for row_index, score in sorted_ndcg_rows[k]:
            table.add_row(str(row_index), f"{score:.4f}")

        console.print(f"Sorted Rows for {k.upper()}")
        console.print(table)
        
        
def calculate_average_ndcg_per_row(ndcg_details):
    average_ndcg_scores = []
    for row in ndcg_details:
        average_score = sum(row[f'NDCG@{k}'] for k in range(1, 6)) / 5
        average_ndcg_scores.append(average_score)
    return average_ndcg_scores

def display_sorted_average_ndcg_rows(average_ndcg_scores):
    sorted_average_scores = sorted(enumerate(average_ndcg_scores), key=lambda x: x[1], reverse=True)
    
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Row Index", style="dim")
    table.add_column("Average NDCG Score", justify="right")

    for row_index, score in sorted_average_scores:
        table.add_row(str(row_index), f"{score:.4f}")

    console.print("Rows Sorted by Average NDCG Score (Best to Worst)")
    console.print(table)
    
recommendations = [
    ["Attribute Based Credentials", "Psuedonymous Identity", "Location Granularity", 
     "Decoupling content and location information visibility", "Onion Routing"],
    ["Psuedonymous Identity", "Attribute Based Credentials", "Obtaining Explicit Consent", 
     "Onion Routing", "Protection against Tracking"],
    ["Support Selective Disclosure", "Pseudonymous Messaging", "Psuedonymous Identity", 
     "Attribute Based Credentials", "Added-noise measurement obfuscation"],
    ["Attribute Based Credentials", "Active broadcast of presence", "Added-noise measurement obfuscation", 
     "Awareness Feed", "Personal Data Store"],
    ["Strip Invisible Metadata", "Added-noise measurement obfuscation", "Lawful Consent", 
     "Obtaining Explicit Consent", "Attribute Based Credentials"],
    ["Active broadcast of presence", "Privacy dashboard", "Attribute Based Credentials", 
     "Abridged Terms and Conditions", "Dynamic Privacy Policy Display"],
    ["Lawful Consent", "Attribute Based Credentials", "Awareness Feed", 
     "Selective access control", "Informed Implicit Consent"]
]

recommendations_ideal = [
     ["Location Granularity", "Decoupling content and location information visibility", "Psuedonymous Identity", "Attribute Based Credentials", "O"],
     ["Protection against Tracking", "Psuedonymous Identity", "Attribute Based Credentials", "Onion Routing", "O"],
     ["Support Selective Disclosure", "Pseudonymous Messaging", "Psuedonymous Identity", 
     "Attribute Based Credentials", "Added-noise measurement obfuscation"],
     ["Attribute Based Credentials", "Added-noise measurement obfuscation", "Active broadcast of presence", "Awareness Feed", "Personal Data Store"],
    ["Strip Invisible Metadata", "Added-noise measurement obfuscation", "O", 
     "O", "Attribute Based Credentials"],
    ["Active broadcast of presence", "Abridged Terms and Conditions", "Privacy dashboard", "Dynamic Privacy Policy Display", "O"],
    ["Lawful Consent", "Awareness Feed", 
     "Informed Implicit Consent", "Selective access control", "O"]
]

# Example usage
metrics = calculate_metrics(recommendations, recommendations_ideal)
sorted_ndcg_rows = sort_ndcg_rows(metrics['ndcg_details'])
best_ndcg_rows = find_best_ndcg_rows(metrics['ndcg_details'])

print("Metrics:", metrics)
print("==="*5)
print("Best NDCG rows:", best_ndcg_rows)
print("==="*5)
display_sorted_ndcg_rows(sorted_ndcg_rows)

average_ndcg_scores = calculate_average_ndcg_per_row(metrics['ndcg_details'])

# Print using rich console
display_sorted_average_ndcg_rows(average_ndcg_scores)