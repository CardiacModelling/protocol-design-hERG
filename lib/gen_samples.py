"""
Try using latin hypercube sampling for brute force protocol design.

This generate samples based on the parameters values from `prior-parameters`.
"""

def sample(boundaries, n=1000):
    # return boundaries.sample(n)
    import lhs

    unit_samples = lhs.lhs(
            boundaries.n_parameters(), samples=n, criterion='centermaximin')

    # transform to within boundaries
    samples = boundaries.lower() + unit_samples * boundaries.range()

    return samples

