import unittest

try:
    import torch
except ImportError:  # The local CPU-only authoring environment intentionally lacks torch.
    torch = None


def _random_corner(L, H, gen):
    """A point of the box that pushes each free coordinate to a random endpoint."""
    u = torch.rand(L.shape, generator=gen, dtype=L.dtype)
    return torch.where(u < 0.5, L, H)


@unittest.skipIf(torch is None, "torch is not installed in the local authoring environment")
class CollisionBoxTests(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(0)
        self.W = torch.randn(24, 256, generator=self.gen, dtype=torch.float64) * 0.05
        # heavy-tailed activation maxima, as in real text embeddings
        self.a = torch.rand(256, generator=self.gen, dtype=torch.float64) ** 4 * 30 + 0.1

    def test_w8a8_box_is_an_exact_deployed_collision_after_recalibration(self):
        from lib.collision import w8a8_box, w8a8_dequant
        L, H = w8a8_box(self.W, self.a)
        d0, c0 = w8a8_dequant(self.W, self.a)
        for _ in range(5):
            Ws = self.W + _random_corner(L, H, self.gen)
            d1, c1 = w8a8_dequant(Ws, self.a)
            self.assertTrue(torch.equal(c0, c1))
            torch.testing.assert_close(d1, d0, rtol=1e-12, atol=0)

    def test_without_column_caps_recalibration_breaks_codes(self):
        from lib.collision import w8a8_box, w8a8_dequant
        L, H = w8a8_box(self.W, self.a, column_cap=False)
        _, c0 = w8a8_dequant(self.W, self.a)
        Ws = self.W + _random_corner(L, H, self.gen)
        _, c1 = w8a8_dequant(Ws, self.a)
        self.assertLess(float((c0 == c1).double().mean()), 1.0)

    def test_weight_only_boxes_are_exact(self):
        from lib.collision import wo_box, wo_dequant
        W = torch.randn(8, 256, generator=self.gen, dtype=torch.float64)
        for q in ("int8_wo", "int4_g128", "int4_g64", "nf4_g64"):
            L, H = wo_box(W, q)
            d0, c0 = wo_dequant(W, q)
            d1, c1 = wo_dequant(W + _random_corner(L, H, self.gen), q)
            self.assertTrue(torch.equal(c0, c1), q)
            torch.testing.assert_close(d1, d0, rtol=1e-12, atol=0)

    def test_solver_certificate_brackets_the_optimum(self):
        from lib.collision import solve
        E = torch.randn(40, 16, generator=self.gen, dtype=torch.float64)
        G = E.t() @ E
        Delta = torch.randn(3, 16, generator=self.gen, dtype=torch.float64)
        L, H = -0.2 * torch.ones_like(Delta), 0.3 * torch.ones_like(Delta)
        X, f, lb, base, _ = solve(Delta, G, L, H, max_iter=20000, gap_tol=1e-10)
        self.assertLessEqual(lb, f + 1e-9)
        self.assertLess(f - lb, 1e-6 * base)
        self.assertTrue(bool(((X >= L - 1e-12) & (X <= H + 1e-12)).all()))


if __name__ == "__main__":
    unittest.main()
