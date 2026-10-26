import unittest

import numpy as np
import torch

from knn_dta_reference import TrainingStore, nearest
from run_knn_shared import Backbone, fit_gate, head_predict, weighted_stats


class Checks(unittest.TestCase):
    def test_training_weighted_normalization(self):
        features = np.array([[1., 2.], [3., 6.], [100., 200.]], dtype=np.float32)
        indices = np.array([0, 0, 1])
        mean, std = weighted_stats(features, indices)
        np.testing.assert_allclose(mean, features[indices].mean(0), rtol=1e-6)
        np.testing.assert_allclose(std, features[indices].std(0), rtol=1e-6)
        features[2] = -10000.
        mean2, std2 = weighted_stats(features, indices)
        np.testing.assert_array_equal(mean, mean2)
        np.testing.assert_array_equal(std, std2)

    def test_factorized_search_matches_direct_concatenation(self):
        torch.manual_seed(9)
        d = torch.randn(4, 7, dtype=torch.float64)
        p = torch.randn(3, 5, dtype=torch.float64)
        di = torch.tensor([0, 1, 2, 3, 1, 3, 0])
        pi = torch.tensor([0, 1, 2, 0, 2, 1, 2])
        y = torch.arange(7, dtype=torch.float64)
        store = TrainingStore(d[di], p[pi], y, di.tolist(), pi.tolist(), list(range(7)), ["train"] * 7)
        qd, qp = torch.randn(5, 7, dtype=torch.float64), torch.randn(5, 5, dtype=torch.float64)
        efficient = store.retrieve(qd, qp, 5, 4, 3).labels
        direct = nearest(torch.cat([qd, qp], 1), torch.cat([d[di], p[pi]], 1), y, 5)
        torch.testing.assert_close(efficient.indices, direct.indices)
        torch.testing.assert_close(efficient.distances, direct.distances, rtol=1e-12, atol=1e-12)

    def test_cached_head_reproduces_full_backbone(self):
        torch.manual_seed(3)
        d, p = torch.randn(13, 4), torch.randn(13, 6)
        config = {"entity_hidden": 8, "latent_dim": 5, "head_hidden": 7, "dropout": .1}
        stats = (np.zeros(4, np.float32), np.ones(4, np.float32), np.zeros(6, np.float32), np.ones(6, np.float32))
        model = Backbone(4, 6, config, stats, 7., 2.).eval()
        with torch.no_grad():
            full = model(d, p)
            cached = head_predict(model.head, model.encode_drug(d), model.encode_protein(p), batch=5)
        torch.testing.assert_close(full, cached, rtol=1e-6, atol=1e-6)

    def test_gate_is_invariant_to_label_translation(self):
        rng = np.random.default_rng(7)
        similarity = rng.uniform(.1, .9, 100)
        true = rng.normal(7., 1., 100)
        deep = true + rng.normal(0, .5, 100)
        prior = true + rng.normal(0, .8, 100)
        a = fit_gate(deep, prior, similarity, true)
        b = fit_gate(deep + 5., prior + 5., similarity, true + 5.)
        np.testing.assert_allclose([a["a"], a["b"]], [b["a"], b["b"]], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
