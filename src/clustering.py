import time

try:
    import torch
except ImportError:
    raise ImportError("The torch package is required for this module.")

try:
    from matplotlib import pyplot as plt
except ImportError:
    raise ImportError("The matplotlib package is required for this module.")

try:
    from pykeops.torch import LazyTensor
except ImportError:
    raise ImportError("The pykeops package is required for this module.")


class Clustering:
    @staticmethod
    def KMeans(
        x: torch.Tensor,
        K: int = 10,
        Niter: int = 10,
        verbose: bool = True,
        device: str = "cpu",
    ):
        """
        Perform K-means clustering on the given data

        Parameters
        ----------
        x : torch.Tensor
            The input data for clustering
        K : int, optional
            The number of clusters to form, by default 10
        Niter : int, optional
            The number of iterations for the K-means algorithm, by default 10
        verbose : bool, optional
            Whether to print information about the process, by default True
        device : str, optional
            The device to perform computations on, by default "cpu"

        Returns
        -------
        tuple of torch.Tensor
            The cluster assignments and the cluster centroids
        """

        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor."
        assert x.ndim == 2, "x must be a 2D tensor."
        assert isinstance(K, int), "K must be an integer."
        assert isinstance(Niter, int), "Niter must be an integer."
        assert isinstance(verbose, bool), "verbose must be a boolean."

        dtype = x.dtype
        x = x.to(device)
        start = time.time()
        N, D = x.shape

        assert (
            K <= N
        ), "Number of clusters (K) can't be greater than the number of data points (N)."

        perm = torch.randperm(N).to(device)
        c = x[perm[:K], :].clone()

        x_i = LazyTensor(x.view(N, 1, D))
        c_j = LazyTensor(c.view(1, K, D))

        for _ in range(Niter):
            D_ij = ((x_i - c_j) ** 2).sum(-1)
            cl = D_ij.argmin(dim=1).long().view(-1)

            c.zero_()
            c.scatter_add_(0, cl[:, None].repeat(1, D), x)

            Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
            c /= Ncl

        if verbose is True:
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            print(
                f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
            )
            print(
                f"Timing for {Niter} iterations: {(end - start):.5f}s = {Niter} x {(end - start) / Niter:.5f}s\n"
            )

        return cl, c

    @staticmethod
    def BisectingKMeans(
        x: torch.Tensor,
        K: int = 15,
        Niter: int = 10,
        verbose: bool = True,
        device: str = "cpu",
    ):
        """
        Perform Bisecting K-means clustering on the given data

        Parameters
        ----------
        x : torch.Tensor
            The input data for clustering
        K : int, optional
            The number of clusters to form, by default 15
        Niter : int, optional
            The number of iterations for the Bisecting K-means algorithm, by default 10
        verbose : bool, optional
            Whether to print information about the process, by default True
        device : str, optional
            The device to perform computations on, by default "cpu"

        Returns
        -------
        tuple of torch.Tensor
            The cluster assignments and the cluster centroids
        """
        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor."
        assert x.ndim == 2, "x must be a 2D tensor."
        assert isinstance(K, int), "K must be an integer."
        assert isinstance(Niter, int), "Niter must be an integer."
        assert isinstance(verbose, bool), "verbose must be a boolean."

        dtype = x.dtype
        x = x.to(device)
        start = time.time()
        N, D = x.shape

        clusters = torch.zeros(N, dtype=torch.long, device=device)

        for i in range(K - 1):
            sse = torch.zeros(i + 1, dtype=dtype, device=device)
            for j in range(i + 1):
                mask = clusters == j
                if mask.sum() > 0:
                    sse[j] = ((x[mask] - x[mask].mean(dim=0)) ** 2).sum()

            max_sse_cluster = sse.argmax()

            mask = clusters == max_sse_cluster
            new_clusters, _ = Clustering.KMeans(
                x[mask], K=2, Niter=Niter, verbose=verbose, device=device
            )
            clusters[mask] = new_clusters + i + 1

        c = torch.zeros((K, D), dtype=dtype, device=device)
        for i in range(K):
            mask = clusters == i
            if mask.sum() > 0:
                c[i] = x[mask].mean(dim=0)

        if verbose is True:
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            print(
                f"Bisecting K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
            )
            print(
                f"Timing for {Niter} iterations: {(end - start):.5f}s = {Niter} x {(end - start) / Niter:.5f}s\n"
            )

        return clusters, c

    @staticmethod
    def MiniBatchKMeans(
        x: torch.Tensor,
        K: int = 8,
        Niter: int = 100,
        batch_size: int = 1024,
        verbose: bool = True,
        device: str = "cpu",
        reassignment_ratio: float = 0.01,
    ):
        """
        Perform MiniBatch K-means clustering on the given data

        Parameters
        ----------
        x : torch.Tensor
            The input data for clustering
        K : int, optional
            The number of clusters to form, by default 8
        Niter : int, optional
            The number of iterations for the MiniBatch K-means algorithm, by default 100
        batch_size : int, optional
            Size of the mini batches, by default 1024
        verbose : bool, optional
            Whether to print information about the process, by default False
        device : str, optional
            The device to perform computations on, by default "cpu"
        reassignment_ratio : float, optional
            Control the fraction of the maximum number of counts for a center to be reassigned

        Returns
        -------
        tuple of torch.Tensor
            The cluster assignments and the cluster centroids
        """

        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor."
        assert x.ndim == 2, "x must be a 2D tensor."
        assert isinstance(K, int), "K must be an integer."
        assert isinstance(Niter, int), "Niter must be an integer."
        assert isinstance(verbose, bool), "verbose must be a boolean."

        dtype = x.dtype
        x = x.to(device)
        start = time.time()
        N, D = x.shape

        assert (
            K <= N
        ), "Number of clusters (K) can't be greater than the number of data points (N)."

        perm = torch.randperm(N).to(device)
        c = x[perm[:K], :].clone()

        for _ in range(Niter):
            batch_indices = torch.randint(0, N, (batch_size,))
            batch = x[batch_indices]

            x_i = LazyTensor(batch.view(batch_size, 1, D))
            c_j = LazyTensor(c.view(1, K, D))

            D_ij = ((x_i - c_j) ** 2).sum(-1)
            cl = D_ij.argmin(dim=1).long().view(-1)

            Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
            new_c = torch.zeros_like(c)
            new_c.scatter_add_(0, cl[:, None].repeat(1, D), batch)
            c = (1 - reassignment_ratio) * c + reassignment_ratio * new_c / Ncl

        x_i = LazyTensor(x.view(N, 1, D))
        D_ij = ((x_i - c_j) ** 2).sum(-1)
        cl = D_ij.argmin(dim=1).long().view(-1)

        if verbose is True:
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            print(
                f"Mini-batch K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
            )
            print(
                f"Timing for {Niter} iterations: {(end - start):.5f}s = {Niter} x {(end - start) / Niter:.5f}s\n"
            )

        return cl, c

    @staticmethod
    def plot_clusters(x: torch.Tensor, cl: torch.Tensor, c: torch.Tensor):
        """
        Plot the clusters and centroids obtained from the clustering algorithm.

        Parameters
        ----------
        x : torch.Tensor
            The input data used for clustering.
        cl : torch.Tensor
            The cluster assignments for each data point.
        c : torch.Tensor
            The cluster centroids.

        Returns
        -------
        None
            The function displays the plot of the clusters and centroids.
        """
        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor."
        assert x.ndim == 2, "x must be a 2D tensor."
        assert isinstance(cl, torch.Tensor), "cl must be a torch.Tensor."
        assert cl.ndim == 1, "cl must be a 1D tensor."
        assert isinstance(c, torch.Tensor), "c must be a torch.Tensor."
        assert c.ndim == 2, "c must be a 2D tensor."

        plt.figure(figsize=(8, 8))
        plt.scatter(
            x[:, 0].cpu(),
            x[:, 1].cpu(),
            c=cl.cpu(),
            s=30000 / len(x),
            cmap="nipy_spectral",
        )
        plt.scatter(c[:, 0].cpu(), c[:, 1].cpu(), c="black", s=50, alpha=0.8)
        plt.axis([-2, 2, -2, 2])
        plt.tight_layout()
        plt.show()
