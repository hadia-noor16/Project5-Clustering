from __future__ import annotations
from typing import Optional
import numpy as np


class FluxEquilibriumClustering:
    """Flux Equilibrium Clustering (FEC) with improved defaults.

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbours when building the k-NN graph.
        For better clustering, consider k in range [8, 15].
    sigma : Optional[float], default=None
        Gaussian kernel bandwidth for the initial edge weights. If ``None``,
        infer from data. Recommended: use a fraction of median pairwise distance
        (e.g., 0.2 * median_distance) for more discriminative weights.
    alpha : float, default=0.5
        Fraction of stored flux that can be redistributed at each iteration.
        Higher values (0.5-0.8) promote more aggressive clustering.
    T : int, default=25
        Number of transport iterations. Consider 30-50 for better convergence.
    epsilon : float, default=1e-4
        Threshold on the *outgoing* flux below which a node is considered a sink.
        Lower values (1e-4 to 1e-6) help create fewer, larger clusters.
    beta : Optional[float], default=None
        Controls the non-linear reinforcement when re-weighting edges.

        * ``beta is None``  –  use ``phi(f) = log(1+f)`` (recommended).
        * ``beta = 0``      –  recovers the *original* FEC behaviour.
        * any other value   –  use ``phi(f) = f**beta``.

    Notes
    -----
    Implementation hints for better clustering:
    - Handle small datasets: ensure k <= n-1 to avoid boundary errors
    - Consider relaxed downhill flow: allow flux to similar-distance neighbors
    - Add cycle detection in path-following to prevent infinite loops
    - Post-process: merge very small clusters with nearest larger ones
    """

    def __init__(
            self,
            k: int = 10,
            sigma: Optional[float] = None,
            alpha: float = 0.5,
            T: int = 25,
            epsilon: float = 1e-4,
            beta: Optional[float] = None,
    ) -> None:

        self.k=k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta

        # Hint: self.k = k, self.sigma = sigma, etc.
        #raise NotImplementedError

    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute cluster labels for *X*.

        Implementation steps:
        1. Compute global centroid and distances to centroid
        2. Build symmetric k-NN graph with Gaussian edge weights
           - Handle small datasets: use effective_k = min(k, n-1)
           - If sigma is None, infer from data (e.g., fraction of median distance)
        3. Initialize flux: f_i = 1 for all nodes
        4. Run T flux-transport iterations:
           - Apply flux-aware reweighting (beta parameter)
           - Use relaxed downhill criterion (optional improvement)
           - Update flux: f_i(t+1) = f_i(t) - outgoing + incoming
        5. Identify sinks: nodes with outgoing flux < epsilon
        6. Assign clusters via steepest downhill paths
           - Add cycle detection to prevent infinite loops
           - Consider post-processing to merge small clusters

        The final labels must be stored in ``self.labels``.
        """
        # Hint: Start with computing centroid = X.mean(axis=0)

        # Add your code here
    # def fit(self, X) -> "FluxEquilibriumClustering":

        #STEP 1 (centroid and distance of each point from centroid)

        centroid = np.mean(X, axis=0)  #axis=0 means avg of all x-values, avg of all y-values

        #print(centroid)
                                    # axis=1 means first-neighbor column and second-neighbor column separately
        distances=np.linalg.norm(X-centroid, axis=1)   # distance from each point to centroid. We don't need only vector but sqrt(x^2 + y^2)
                                                    # axis=1 means calculate one distance for each row of x
        #print(distances)

        #Step 2 (pair wise squared distance)

        #bubble sorting loop, p0 is compared with all points including itself, then loop is increments to p1 and then p1 will be compared with every point in array

        pw_squared_distance=[]

        for i in range(len(X)):    # iteration1:  for P0 in array for the length of array
            rows=[]

            for j in range(len(X)):    # for all P0-P7
                rows.append(np.sum((X[i]-X[j])**2))    # when p0 is compared with p0, the difference will be 0, then p0 will be compared with p1.,p2 and so on....

            pw_squared_distance.append(rows)          # append all distances when p0 is compared with p0-p7,

        pw_sq=np.array(pw_squared_distance)      # convert it into array

       # print(pw_sq)

        # Step 3 (Calculating sigma)

        pairwise_distances=[]

        for i in range(len(X)):
            for j in range(i+1, len(X)):
                sq_dist=np.sum((X[i]-X[j])**2)
                reg_dist=np.sqrt(sq_dist)
                pairwise_distances.append(reg_dist)

        median_dist = np.median(pairwise_distances)

        if self.sigma is None:
            sigma = 0.2 * median_dist
        else:
            sigma = self.sigma

        sigma_sq = sigma * sigma

        #print(f"sigma_squared: {sigma_sq}")

        # Step 4 (Calculate two nearest neighbors from  each row in pairwise distances ignoring self compare)

        nearest=[]

        wij=[]
        effective_k = min(self.k, len(X) - 1)

        for i in range(len(pw_sq)):

            nearest.append(
                np.argsort(pw_sq[i])[1:effective_k + 1]
            )
            #nearest.append( np.argsort(pw_sq[i])[1:3])   # range 1-3 will leave element at index 0 (which will always be 0, because 0 is the smallest, so in argsort 0 comes first)
                                                         # will pick only 1,2 index values.
        #print(np.array(nearest))


        # Step 5 ( calculating Using the Gaussian kernel, the resulting k-NN edge weights for each node)

        weights = []
        for i in range(len(nearest)):   # for first iteration, i=0, nearest[0] == [3,2]
            row_weights=[]
            for y in range(len(nearest[i])):   # for first iteration, y in nearest[0][0] ==> 3
                j=nearest[i][y]                # j= 3

                sq_dist=pw_sq[i][j]          # from sample 0 to sample 3, then in 2nd iteration sample 0 to sample 2
                wij=np.exp(-sq_dist/sigma_sq)

                row_weights.append(wij)
            weights.append(row_weights)
        w_ij = np.array(weights)

        #print(np.array(weights))

        # Step 6 (Initialize flux)

        flux=np.ones(len(X))    # initialize flux with all ones
        #delta_incoming=np.zeros(len(X))


        tau= 0.1 * sigma
        for x in range(self.T):    # if T=5, means how many times we want to distribute flux, loop runs 0-4
            delta_final_out = []
            delta_incoming = np.zeros(len(X))  # array of 8 zeroes in the beginning for incoming flux
            for i in range (len(X)):    # for sample 0

                # calculate sum_weights
                sum_weights=0           # sum_weights is 0 initialized
                for y in range(len(nearest[i])):    # y = nearest[0] that is [3,2]
                    j=nearest[i][y]                # nearest[0][0] == 3, so j=3
                                                # j is the sample receiving the flux, in this case 0 is giving away flux to sample 3 and 2

                    if distances[j] <= distances[i] + tau:    # if sample 3 <= sample[0] + tau, we can add both the samples weight

                        # Edge-weight will be different in each iteration, so Computing flux-aware edge weight

                        if x==0 or self.beta==0:
                            phi =1
                        elif self.beta is None:
                            phi= np.log (1+ flux[j])
                        else:
                            phi= flux[j] ** self.beta

                        adjusted_weight = weights[i][y] * phi
                        sum_weights += adjusted_weight       # sum_weights= weights[0][0] = 0.447

                                                    # in next iteration y will be 1, so weights [0][1]= 0.4090
                        #flux= self.alpha * (weights[i][y]/sum_weights)

                # calculate delta-outgoing, and incoming
                delta_out=[]

                for y in range(len(nearest[i])):
                    j=nearest[i][y]    # j = 3 and 2 in each iteration respectively

                    if distances[j] <= distances[i] + tau:

                        # Edge-weight will be different in each iteration, so Computing flux-aware edge weight

                        if x == 0 or self.beta == 0:
                            phi = 1
                        elif self.beta is None:
                            phi = np.log(1 + flux[j])
                        else:
                            phi = flux[j] ** self.beta

                        adjusted_weight = weights[i][y] * phi

                        delta=  self.alpha * adjusted_weight * flux[i] / sum_weights   #outgoing from sample 0, becomes incoming for sample 2 and 3
                        delta_incoming[j]+=delta                  # so for first iteration j=3, means sample 3 receives this flux from sample 0
                    else:
                        delta=0      # if downhill condition fails, it won't distribute flux so outgoing from this sample will be 0

                    delta_out.append(delta)
                delta_final_out.append(delta_out)  # how much sample i sent to each neighbor

        #print(np.array(delta_final_out))
            outgoing= np.sum(delta_final_out, axis=1)   #total amount sample i sent
            incoming= delta_incoming
            #print(f"incoming: {incoming}")

          #  print(f"outgoing: {np.array(outgoing)}")

            new_flux= flux - outgoing + incoming

            flux=new_flux
           # print(np.sum(flux))
        # Detect sinks using outgoing flux from the final iteration

        sinks=[]

        for i in range(len(outgoing)):
            if outgoing[i] < self. epsilon:
                sinks.append(i)

        if len(sinks)==0:
            closest = np.argmin(distances)   # if there is no sink node, choose the closest sample from centroid as the sink node
            sinks.append(closest)

        #print(sinks)


        # detect sink and non-sink nodes and label sink nodes

        labels= np.full(len(X), -1)

        for label,sink in enumerate(sinks):   # gives position number and sink value position (0,1,2) value= [2,5,7]
            labels[sink]= label

        #print(labels)

        # finding steepest neighbors of non-sink nodes

        steepest={}

        for i in range(len(X)):
            if i not in sinks:
                max_node= np.argmax(delta_final_out[i])
                steepest[i]= int(nearest[i][max_node])
        #print(np.array(steepest))


        # detecting a cycle and breaking it by creating current node a sink node
        for i in range(len(X)):  # 0--7
            visited=[]
            node=i    # node=0

            while node not in sinks:   # 0 is not in sinks? true
                if node in visited:  # 0 is not in visited yet
                    sinks.append(int(node))  # if node is already in visited, make it a new sink
                    labels[node] = len(sinks) - 1  # label the sink, since this is new sink
                    break
                visited.append(node)   # it goes  ahead and add 0 to visited
                node=steepest[node]    # node's new value becomes steepest[0], and steepest neighbor of 0 is 3, so node becomes 3
            sink_label = labels[node]   # after the path ends, give this new sink a cluster label

            for node in visited:
                labels[node]=sink_label  # give that same label to every visited node leading to that sink

        #print("labels BEFORE merging:", labels)
        #print("sinks:", sinks)
        #print("steepest:", steepest)


        # merging small cluster
        s_min = max(2, len(X) // 20)

        unique_labels, counts = np.unique(labels, return_counts=True)
        #print(unique_labels)
        #print(counts)

        small_clusters=[]
        large_clusters=[]

        for label, count in zip(unique_labels, counts):
            if count < s_min:
                small_clusters.append(label)
            else:
                large_clusters.append(label)

        #print(f"large_cluster:{large_clusters}")
        #print(f"small_clusters:{small_clusters}")

        for small_label in small_clusters:  # looping through all small clusters
            small_points = X[labels == small_label]  # get points that belong to that small cluster
            small_centroid = np.mean(small_points, axis=0)  # centroid of small points

            closest_distance = float("inf")  # intitialzing closest_dist with infinity,
            closest_label = None

            for large_label in large_clusters:
                large_points = X[labels == large_label]
                large_centroid = np.mean(large_points, axis=0)

                dist_clusters=np.linalg.norm(small_centroid-large_centroid)

                if dist_clusters < closest_distance:   # upon checking first dist, it will be smaller than infinity
                    closest_distance = dist_clusters
                    closest_label = large_label

            labels[labels == small_label] = closest_label

        #print(labels)


    # ---- DO NOT CHANGE THE REST OF THIS METHOD ---- #
    # We require you maintain these attributes for testing purpose.
        self.w_ij = w_ij
        self.flux = flux
        sink_mask = np.zeros(len(X), dtype=bool)
        for sink in sinks:
            sink_mask[sink] = True
        self.sinks = sink_mask
        self.labels = labels

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that returns the cluster labels."""
        self.fit(X)
        return self.labels

    def __repr__(self) -> str:
        params = (
            f"k={getattr(self, 'k', '?')}",
            f"sigma={getattr(self, 'sigma', '?')}",
            f"alpha={getattr(self, 'alpha', '?')}",
            f"T={getattr(self, 'T', '?')}",
            f"epsilon={getattr(self, 'epsilon', '?')}",
            f"beta={getattr(self, 'beta', '?')}",
        )
        return f"{self.__class__.__name__}({', '.join(params)})"

# #X = np.array([
#         [2.67, 2.95],
#         [1.29, 4.31],
#         [1.57, 2.33],
#         [3.40, 2.00],
#         [-3.85, -3.73],
#         [-2.56, -1.17],
#         [-2.97, -3.89],
#         [-2.57, -3.48]
#     ])
#
# model = FluxEquilibriumClustering()
# model.fit(X)