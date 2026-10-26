"""Numerical and isolation checks independent of benchmark outcomes."""
import unittest

import torch
from torch import nn

from knn_dta_reference import AdaptiveAggregation, FixedParameters, Mix, TrainingStore, fixed_prediction, nearest


class LinearHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, drug, protein):
        return self.scale * drug[:, 0] + protein[:, 0]


class Checks(unittest.TestCase):
    def setUp(self):
        self.drug = torch.tensor([[0.], [2.], [0.]])
        self.protein = torch.tensor([[0.], [0.], [3.]])
        self.labels = torch.tensor([1., 5., 7.])
        self.store = TrainingStore(self.drug, self.protein, self.labels,
                                   ["d0", "d1", "d0"], ["p0", "p0", "p1"],
                                   [1, 2, 3], ["train"] * 3)
        self.query = self.store.retrieve(torch.tensor([[1.]]), torch.tensor([[0.]]), 3, 3, 3)
        self.head = LinearHead().eval()

    def test_unique_entity_datastores_and_small_k(self):
        self.assertEqual(len(self.store.drug), 2)
        self.assertEqual(len(self.store.protein), 2)
        self.assertEqual(self.query.drug_neighbors.distances.shape, (1, 2))

    def test_training_only_and_consistent_embeddings(self):
        for splits in (["train", "val", "train"], ["train", "test", "train"]):
            with self.assertRaises(ValueError):
                TrainingStore(self.drug, self.protein, self.labels, ["a", "b", "a"],
                              ["x", "x", "y"], [1, 2, 3], splits)
        with self.assertRaises(ValueError):
            TrainingStore(self.drug, self.protein, self.labels, ["a", "a", "a"],
                          ["x", "x", "y"], [1, 2, 3], ["train"] * 3)

    def test_distance_direction_ties_and_chunking(self):
        self.assertEqual(self.query.labels.indices.tolist(), [[0, 1, 2]])
        torch.testing.assert_close(self.query.labels.distances, torch.tensor([[1., 1., 10.]]))
        q = torch.tensor([[1.], [0.], [3.]])
        a = nearest(q, self.drug, self.labels, 3, chunk_size=1)
        b = nearest(q, self.drug, self.labels, 3, chunk_size=3)
        torch.testing.assert_close(a.indices, b.indices)
        torch.testing.assert_close(a.distances, b.distances)

    def test_all_base_weights_recover_head_exactly(self):
        mix = Mix(2, 10., 1.)
        value = fixed_prediction(self.query, self.head, FixedParameters(mix, mix, mix))
        torch.testing.assert_close(value, torch.tensor([2.]), rtol=0, atol=0)

    def test_label_aggregation_hand_calculation(self):
        # The first two pairs are equidistant, so their mean is (1 + 5) / 2.
        base, label = Mix(2, 10., 1.), Mix(2, 10., 0.)
        value = fixed_prediction(self.query, self.head, FixedParameters(label, base, base))
        torch.testing.assert_close(value, torch.tensor([3.]))
        # Near-zero temperature must favor the closest pair, not the farthest.
        query = self.store.retrieve(torch.tensor([[0.]]), torch.tensor([[0.]]), 3, 2, 2)
        value = fixed_prediction(query, self.head, FixedParameters(Mix(3, .001, 0.), base, base))
        torch.testing.assert_close(value, torch.tensor([1.]))

    def test_feature_aggregation_hand_calculation(self):
        # One drug neighbor is zero; protein is unchanged. With half drug mixing,
        # the head receives drug=0.5 and protein=0, giving 2*0.5=1.
        value = fixed_prediction(self.query, self.head,
                                 FixedParameters(Mix(2, 10., 1.), Mix(1, 10., .5), Mix(2, 10., 1.)))
        torch.testing.assert_close(value, torch.tensor([1.]))

    def test_adaptive_uniform_weights_and_frozen_head_gradients(self):
        torch.manual_seed(7)
        model = AdaptiveAggregation(self.head, 3, 2, 2)
        for network in (model.label_network, model.drug_network, model.protein_network):
            for parameter in network.parameters():
                nn.init.zeros_(parameter)
        model.train()
        self.assertFalse(model.head.training)
        value = model(self.query)
        # Drug mean=(query 1 + neighbors 0+2)/3=1; protein mean=(0+0+3)/3=1.
        # Head=3; label aggregation=(3+1+5+7)/4=4.
        torch.testing.assert_close(value, torch.tensor([4.]))
        value.square().mean().backward()
        self.assertIsNone(model.head.scale.grad)
        for network in (model.label_network, model.drug_network, model.protein_network):
            self.assertGreater(float(network[-1].bias.grad.abs().sum()), 0.)

    def test_datastore_detaches_encoder_graph(self):
        self.assertFalse(self.query.labels.values.requires_grad)
        self.assertFalse(self.query.drug_neighbors.values.requires_grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not visible in this process")
    def test_cpu_gpu_parity(self):
        store = TrainingStore(self.drug.cuda(), self.protein.cuda(), self.labels.cuda(),
                              ["d0", "d1", "d0"], ["p0", "p0", "p1"], [1, 2, 3], ["train"] * 3)
        query = store.retrieve(self.query.drug.cuda(), self.query.protein.cuda(), 3, 3, 3)
        parameters = FixedParameters(Mix(3, 10., .5), Mix(2, 10., .7), Mix(2, 10., .8))
        cpu = fixed_prediction(self.query, self.head, parameters)
        gpu = fixed_prediction(query, LinearHead().cuda().eval(), parameters)
        torch.testing.assert_close(cpu, gpu.cpu())
        torch.manual_seed(7)
        cpu_model = AdaptiveAggregation(LinearHead().eval(), 3, 2, 2)
        gpu_model = AdaptiveAggregation(LinearHead().cuda().eval(), 3, 2, 2).cuda()
        gpu_model.load_state_dict(cpu_model.state_dict())
        torch.testing.assert_close(cpu_model(self.query), gpu_model(query).cpu())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
