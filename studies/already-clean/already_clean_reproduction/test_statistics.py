"""Small numerical checks for estimators and undefined-vector behavior."""
import unittest

import numpy as np

from .common import birth, cosine, wilson
from .similarity import bootstrap_mean, identity_permutation, normalize_rows, pole_weights, similarity_test
from .followup import layer_bootstrap, retained_indices


class StatisticsTests(unittest.TestCase):
    def test_cosine_zero_is_missing(self):
        self.assertIsNone(cosine([0, 0], [1, 2]))
        self.assertAlmostEqual(cosine([1, 2], [2, 4]), 1)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0)

    def test_birth_requires_persistence(self):
        self.assertEqual(birth([False, True, False, True, True]), 3)
        self.assertEqual(birth([False, False, True]), 2)
        self.assertIsNone(birth([False, False]))

    def test_negative_pole_strength(self):
        np.testing.assert_equal(pole_weights("L01N1-", [(0, -3), (1, -1)]), [3, 1])
        np.testing.assert_equal(pole_weights("L01N1+", [(0, 3), (1, 1)]), [3, 1])
        with self.assertRaises(AssertionError):
            pole_weights("L01N1-", [(0, 3)])

    def test_pair_clustering_changes_unit_count(self):
        config = {"seed": 0, "bootstrap_replicates": 100}
        out = bootstrap_mean([1, 1, 0], ["a", "a", "b"], config)
        self.assertEqual(out["n_neurons"], 2)
        self.assertEqual(out["mean"], .5)

    def test_exact_identity_null_and_exclude_sign_pair(self):
        config = {"seed": 0, "bootstrap_replicates": 100, "permutations": 100}
        neurons = [{"pole": "L01N1+", "group": "random"}, {"pole": "L01N2+", "group": "random"}]
        c = np.repeat(np.eye(2)[:, None, :], 16, axis=1)
        result = identity_permutation(c, neurons, config)
        self.assertEqual(result["null_draws"], 2)
        self.assertEqual(result["p_one_sided"], .5)
        summary, rows = similarity_test(c, neurons, config)
        self.assertEqual(summary["mean"], 1)
        neurons[1]["pole"] = "L01N1-"
        summary, rows = similarity_test(c, neurons, config)
        self.assertEqual(summary["n_neurons"], 0)
        self.assertEqual(rows, [])

    def test_wilson_boundaries(self):
        self.assertEqual(wilson(0, 10)[0], 0)
        self.assertAlmostEqual(wilson(10, 10)[1], 1)

    def test_layer_bootstrap_preserves_neuron_weighted_estimand(self):
        result = layer_bootstrap([1, 1, 0], ["a", "a", "b"], {"seed": 0, "bootstrap_replicates": 100})
        self.assertAlmostEqual(result["mean"], 2 / 3)
        self.assertEqual(result["n_layers"], 2)

    def test_remove_shared_contexts_from_both_sides(self):
        neuron = {"ctx": [["before", "a", "after"], ["x", "b", "y"], ["different", "b", "end"]],
                  "steps": [{"top": [(0, 1), (1, 1)]} for _ in range(16)]}
        neuron["steps"][15] = {"top": [(1, 1), (2, 1)]}
        q, target, _ = retained_indices(neuron, "text")
        self.assertEqual(q, [0])
        self.assertEqual(target, [2])
        q, target, _ = retained_indices(neuron, "token")
        self.assertEqual(q, [0])
        self.assertEqual(target, [])


if __name__ == "__main__":
    unittest.main()
