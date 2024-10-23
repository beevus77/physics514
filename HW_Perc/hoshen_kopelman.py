def hoshen_kopelman(labels, grid):
    """
    Implement the Hoshen-Kopelman algorithm for cluster labeling.

    Args:
    labels (numpy.ndarray): Array to store cluster labels for each site in the grid
    grid (numpy.ndarray): 2D array representing the grid where clusters are identified

    Returns:
    int: The number of clusters found
    """
    # Initialize variables
    current_label = 1
    equivalences = {}

    # First pass: label clusters
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:  # If the site is occupied
                neighbors = []
                if i > 0 and labels[i - 1, j] > 0:  # Check above
                    neighbors.append(labels[i - 1, j])
                if j > 0 and labels[i, j - 1] > 0:  # Check left
                    neighbors.append(labels[i, j - 1])

                if not neighbors:  # No neighbors, new cluster
                    labels[i, j] = current_label
                    current_label += 1
                else:  # Neighbors found
                    min_label = min(neighbors)
                    labels[i, j] = min_label
                    for neighbor in neighbors:
                        if neighbor != min_label:
                            equivalences[neighbor] = min_label

    # Second pass: resolve equivalences
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] > 0:
                original_label = labels[i, j]
                while original_label in equivalences:
                    original_label = equivalences[original_label]
                labels[i, j] = original_label

    return current_label - 1  # Return the number of clusters found

