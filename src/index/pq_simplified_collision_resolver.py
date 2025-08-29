"""
Simplified collision resolution algorithm for Product Quantization (PQ).

This module provides functions to resolve collisions by reassigning items
based on reconstruction error and nearest codebook vector alternatives.
"""

import numpy as np
import copy
from typing import List, Tuple, Dict, Set, Optional


def compute_reconstruction_errors(collision_items: List[int], 
                                all_distances: np.ndarray) -> List[float]:
    """
    Compute reconstruction error for each item in a collision group.
    
    Args:
        collision_items: List of item indices in collision
        all_distances: Array of shape (num_items, num_tokens, num_codebooks)
                      containing distances to each codebook vector
    
    Returns:
        List of reconstruction errors (sum of min distances per token)
    """
    reconstruction_errors = []
    
    for item in collision_items:
        # Sum of minimum distances across all tokens for this item
        error = np.sum([np.min(token_distances) for token_distances in all_distances[item]])
        reconstruction_errors.append(error)
    
    return reconstruction_errors


def select_items_to_reassign(collision_items: List[int], 
                           all_distances: np.ndarray) -> List[int]:
    """
    Select which items need to be reassigned in a collision group.
    Keep the item with smallest reconstruction error, reassign others.
    
    Args:
        collision_items: List of item indices in collision
        all_distances: Array of distances to codebook vectors
    
    Returns:
        List of item indices that need to be reassigned
    """
    if len(collision_items) <= 1:
        return []
    
    reconstruction_errors = compute_reconstruction_errors(collision_items, all_distances)
    
    # Find item with minimum reconstruction error (keep this one)
    min_error_idx = np.argmin(reconstruction_errors)
    keep_item = collision_items[min_error_idx]
    
    # Return all other items for reassignment
    items_to_reassign = [item for item in collision_items if item != keep_item]
    
    # print(f"Collision group of {len(collision_items)} items: keeping item {keep_item} "
    #       f"(error: {reconstruction_errors[min_error_idx]:.4f}), "
    #       f"reassigning {len(items_to_reassign)} items")
    
    return items_to_reassign


def generate_candidate_codes(item_idx: int,
                           original_code: List[int],
                           all_distances: np.ndarray,
                           sort_distances_index: np.ndarray,
                           M: int) -> List[Tuple[float, List[int]]]:
    """
    Generate candidate codes for an item by exploring M nearest codebook vectors
    per token (excluding the current colliding assignment).
    
    Args:
        item_idx: Index of the item to reassign
        original_code: Current code assignment (list of token indices)
        all_distances: Array of distances to codebook vectors
        sort_distances_index: Sorted indices of codebook vectors by distance
        M: Number of nearest codebook vectors to consider per token
    
    Returns:
        List of (distance, new_code) tuples sorted by distance
    """
    candidates = []
    num_tokens = len(original_code)
    
    for token_pos in range(num_tokens):
        current_token = original_code[token_pos]
        
        # Get M nearest codebook indices for this token
        nearest_indices = sort_distances_index[item_idx][token_pos][:M]
        nearest_distances = all_distances[item_idx][token_pos][nearest_indices]

        # print(f"  Token {token_pos}: current={current_token}, nearest={nearest_indices}, distances={nearest_distances}")

        # Skip the current assignment (should be the closest, index 0)
        for rank in range(1, M):  # Start from 1 to skip current assignment
            if rank >= len(nearest_indices):
                break
                
            new_token_idx = nearest_indices[rank]
            distance = nearest_distances[rank]
            
            # Create new code with this token change
            new_code = copy.deepcopy(original_code)
            new_code[token_pos] = int(new_token_idx)
            
            candidates.append((distance, new_code))
    
    # Sort candidates by distance (ascending)
    candidates.sort(key=lambda x: x[0])
    
    return candidates


def try_reassign_item(item_idx: int,
                     original_code: List[int],
                     all_distances: np.ndarray,
                     sort_distances_index: np.ndarray,
                     all_indices_str_set: Set[str],
                     M: int = 10) -> Optional[List[int]]:
    """
    Try to find a new unique code assignment for an item.
    
    Args:
        item_idx: Index of the item to reassign
        original_code: Current code assignment
        all_distances: Array of distances to codebook vectors
        sort_distances_index: Sorted indices of codebook vectors by distance
        all_indices_str_set: Set of already used codes (as strings)
        M: Number of nearest codebook vectors to consider per token
    
    Returns:
        New code assignment if successful, None if no assignment found
    """
    candidates = generate_candidate_codes(
        item_idx, original_code, all_distances, sort_distances_index, M
    )
    
    # print(f"  Generated {len(candidates)} candidate codes for item {item_idx}")
    
    # Try candidates in order of increasing distance
    iteration_count = 0
    for distance, candidate_code in candidates:
        iteration_count += 1
        candidate_str = str(candidate_code)
        
        if candidate_str not in all_indices_str_set:
            # print(f"  Found available code for item {item_idx}: {candidate_code} "
            #       f"(distance: {distance:.4f}) after {iteration_count} iterations")
            # return candidate_code
            if iteration_count > 1:
                print(f"Item {item_idx} took {iteration_count} iterations")
            return candidate_code, iteration_count
    
    # No available assignment found
    print(f"  WARNING: No available code assignment found for item {item_idx} "
          f"after trying all {iteration_count} candidates")
    return None, iteration_count


def resolve_collision_group(collision_items: List[int],
                          all_indices: List[List[int]],
                          all_indices_str: List[str],
                          all_indices_str_set: Set[str],
                          all_distances: np.ndarray,
                          sort_distances_index: np.ndarray,
                          M: int = 10) -> List[int]:
    """
    Resolve collisions for a single collision group.
    
    Args:
        collision_items: List of item indices in collision
        all_indices: List of code assignments (modified in-place)
        all_indices_str: List of code assignments as strings (modified in-place)
        all_indices_str_set: Set of used codes (modified in-place)
        all_distances: Array of distances to codebook vectors
        sort_distances_index: Sorted indices of codebook vectors by distance
        M: Number of nearest codebook vectors to consider per token
        
    Returns:
        List of iteration counts for each reassigned item
    """
    # print(f"\nResolving collision group: {collision_items}")
    
    # Select items that need reassignment
    items_to_reassign = select_items_to_reassign(collision_items, all_distances)
    iteration_counts = []
    
    # Try to reassign each selected item
    for item_idx in items_to_reassign:
        original_code = all_indices[item_idx]
        
        new_code, iteration_count = try_reassign_item(
            item_idx, original_code, all_distances, sort_distances_index,
            all_indices_str_set, M
        )
        iteration_counts.append(iteration_count)
        
        if new_code is not None:
            # Update assignments - this ensures progressive code addition
            all_indices[item_idx] = new_code
            all_indices_str[item_idx] = str(new_code)
            all_indices_str_set.add(str(new_code))  # ← Key: next items will see this new code
            # print(f"  Successfully reassigned item {item_idx}: {original_code} -> {new_code}")
        else:
            print(f"  FAILED to reassign item {item_idx}, keeping original code {original_code}")

    return iteration_counts  # Return after processing ALL items


def resolve_all_collisions_simplified(all_indices: List[List[int]],
                                     all_indices_str: List[str],
                                     all_indices_str_set: Set[str],
                                     all_distances: np.ndarray,
                                     sort_distances_index: np.ndarray,
                                     collision_item_groups: List[List[int]],
                                     M: int = 10) -> None:
    """
    Main function to resolve all collisions using the simplified algorithm.
    
    Args:
        all_indices: List of code assignments (modified in-place)
        all_indices_str: List of code assignments as strings (modified in-place)  
        all_indices_str_set: Set of used codes (modified in-place)
        all_distances: Array of distances to codebook vectors
        sort_distances_index: Sorted indices of codebook vectors by distance
        collision_item_groups: List of collision groups (each group is list of item indices)
        M: Number of nearest codebook vectors to consider per token
    """
    print(f"Starting collision resolution with {len(collision_item_groups)} collision groups")
    print(f"Using M={M} nearest codebook vectors per token")

    iteration_counts_tot = []

    for i, collision_items in enumerate(collision_item_groups):
        # print(f"\n=== Processing collision group {i+1}/{len(collision_item_groups)} ===")
        iteration_counts = resolve_collision_group(
            collision_items, all_indices, all_indices_str, all_indices_str_set,
            all_distances, sort_distances_index, M
        )
        iteration_counts_tot.extend(iteration_counts)

    print(f"Mean iteration count per collision group: {np.mean(iteration_counts_tot):.2f}")

    print(f"\n=== Collision resolution completed ===")
