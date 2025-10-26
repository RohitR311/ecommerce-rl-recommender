"""
Baseline recommendation models (non-RL) for comparison.

These serve as baselines to evaluate whether RL approaches provide value.
"""

import numpy as np
from typing import List, Optional
from collections import Counter


class RandomRecommender:
    """
    Recommends random items from the catalog.

    This is the simplest baseline - any reasonable algorithm should beat this.
    """

    def __init__(self, n_items: int, seed: Optional[int] = None):
        """
        Args:
            n_items: Total number of items in catalog
            seed: Random seed for reproducibility
        """
        self.n_items = n_items
        self.rng = np.random.RandomState(seed)

    def recommend(self, user_id: int, k: int = 10, exclude: Optional[List[int]] = None) -> List[int]:
        """
        Recommend k random items.

        Args:
            user_id: User ID (ignored for random recommender)
            k: Number of items to recommend
            exclude: Items to exclude from recommendations (e.g., already seen)

        Returns:
            List of k recommended item indices
        """
        available_items = list(range(self.n_items))

        if exclude:
            available_items = [i for i in available_items if i not in exclude]

        if len(available_items) < k:
            k = len(available_items)

        return self.rng.choice(available_items, size=k, replace=False).tolist()


class PopularityRecommender:
    """
    Recommends most popular items globally.

    Simple but often surprisingly effective baseline.
    Weakness: No personalization, recommends same items to everyone.
    """

    def __init__(self, n_items: int):
        """
        Args:
            n_items: Total number of items in catalog
        """
        self.n_items = n_items
        self.item_counts = Counter()
        self.popular_items = []

    def fit(self, interactions: List[tuple]):
        """
        Learn item popularity from training data.

        Args:
            interactions: List of (user_id, item_id, reward) tuples
        """
        # Count interactions per item
        for user_id, item_id, reward in interactions:
            # Weight by reward (e.g., purchase counts more than view)
            self.item_counts[item_id] += reward

        # Sort items by popularity
        self.popular_items = [
            item for item, count in self.item_counts.most_common()
        ]

    def recommend(self, user_id: int, k: int = 10, exclude: Optional[List[int]] = None) -> List[int]:
        """
        Recommend top-k most popular items.

        Args:
            user_id: User ID (ignored for popularity recommender)
            k: Number of items to recommend
            exclude: Items to exclude from recommendations

        Returns:
            List of k recommended item indices
        """
        recommendations = []

        for item in self.popular_items:
            if exclude and item in exclude:
                continue
            recommendations.append(item)
            if len(recommendations) >= k:
                break

        # Fill with remaining items if needed
        while len(recommendations) < k:
            for item in range(self.n_items):
                if item not in recommendations and (not exclude or item not in exclude):
                    recommendations.append(item)
                if len(recommendations) >= k:
                    break

        return recommendations[:k]


class UserKNNRecommender:
    """
    Simple user-based collaborative filtering.

    For each user, find K most similar users and recommend items they liked.
    """

    def __init__(self, n_users: int, n_items: int, k_neighbors: int = 10):
        """
        Args:
            n_users: Total number of users
            n_items: Total number of items
            k_neighbors: Number of similar users to consider
        """
        self.n_users = n_users
        self.n_items = n_items
        self.k_neighbors = k_neighbors

        # User-item interaction matrix
        self.interaction_matrix = np.zeros((n_users, n_items))

    def fit(self, interactions: List[tuple]):
        """
        Build user-item interaction matrix.

        Args:
            interactions: List of (user_id, item_id, reward) tuples
        """
        for user_id, item_id, reward in interactions:
            # Binary or weighted
            self.interaction_matrix[user_id, item_id] = reward

    def _compute_similarity(self, user_id: int) -> np.ndarray:
        """
        Compute cosine similarity with all other users.

        Returns:
            Array of similarity scores
        """
        user_vector = self.interaction_matrix[user_id]

        # Cosine similarity
        norms = np.linalg.norm(self.interaction_matrix, axis=1)
        norms[norms == 0] = 1e-10  # Avoid division by zero

        similarities = self.interaction_matrix @ user_vector / (norms * np.linalg.norm(user_vector))

        # Don't include self
        similarities[user_id] = -1

        return similarities

    def recommend(self, user_id: int, k: int = 10, exclude: Optional[List[int]] = None) -> List[int]:
        """
        Recommend items based on similar users' preferences.

        Args:
            user_id: User ID
            k: Number of items to recommend
            exclude: Items to exclude

        Returns:
            List of k recommended item indices
        """
        # Find similar users
        similarities = self._compute_similarity(user_id)
        similar_users = np.argsort(similarities)[-self.k_neighbors:]

        # Aggregate items liked by similar users
        item_scores = np.zeros(self.n_items)
        for similar_user in similar_users:
            sim_weight = similarities[similar_user]
            item_scores += sim_weight * self.interaction_matrix[similar_user]

        # Remove already interacted items
        if exclude:
            for item in exclude:
                item_scores[item] = -1

        # Remove items user has already interacted with
        user_items = np.where(self.interaction_matrix[user_id] > 0)[0]
        item_scores[user_items] = -1

        # Top-k items
        recommendations = np.argsort(item_scores)[-k:][::-1].tolist()

        return recommendations


if __name__ == "__main__":
    # Example usage
    print("Testing baseline recommenders...\n")

    # Simulate some interactions
    n_users, n_items = 100, 500
    interactions = [
        (0, 10, 1.0),
        (0, 20, 1.0),
        (1, 10, 1.0),
        (1, 30, 1.0),
        (2, 10, 1.0),
        (2, 40, 1.0),
    ]

    # Random recommender
    print("1. Random Recommender")
    random_rec = RandomRecommender(n_items, seed=42)
    recs = random_rec.recommend(user_id=0, k=5)
    print(f"   Recommendations for user 0: {recs}\n")

    # Popularity recommender
    print("2. Popularity Recommender")
    pop_rec = PopularityRecommender(n_items)
    pop_rec.fit(interactions)
    recs = pop_rec.recommend(user_id=0, k=5)
    print(f"   Recommendations for user 0: {recs}")
    print(f"   (Item 10 should be popular since all users interacted with it)\n")

    # User KNN recommender
    print("3. User KNN Recommender")
    knn_rec = UserKNNRecommender(n_users, n_items, k_neighbors=2)
    knn_rec.fit(interactions)
    recs = knn_rec.recommend(user_id=0, k=5, exclude=[10, 20])
    print(f"   Recommendations for user 0: {recs}")
    print(f"   (Should recommend items liked by similar users 1 and 2)")
