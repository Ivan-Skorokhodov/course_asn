import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.graph import DOWNSTREAM, PEER, UPSTREAM, load_graph, make_weighted_graph, save_graph


class CSRTests(unittest.TestCase):
    def test_relationships_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "sample.as-rel2.txt"
            src.write_text("# sample\n24|10343|-1|bgp\n24|11537|0|bgp\n", encoding="utf-8")
            save_graph(src, root / "out")
            graph, types, asns = load_graph(root / "out")
            self.assertEqual(graph.nnz, 4)
            self.assertEqual(set(asns.tolist()), {24, 10343, 11537})
            index = {int(asn): i for i, asn in enumerate(asns)}

            def relation(source, dest):
                r, c = index[source], index[dest]
                pos = np.searchsorted(graph.indices[graph.indptr[r]:graph.indptr[r+1]], c)
                return int(types[graph.indptr[r] + pos])

            self.assertEqual(relation(10343, 24), UPSTREAM)
            self.assertEqual(relation(24, 10343), DOWNSTREAM)
            self.assertEqual(relation(24, 11537), PEER)
            weighted = make_weighted_graph(graph, types, 2)
            self.assertEqual(weighted[index[10343], index[24]], 3)
            self.assertEqual(weighted[index[24], index[10343]], 1)

    def test_conflicting_duplicate_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "bad.txt"
            src.write_text("24|10343|-1|bgp\n24|10343|0|bgp\n")
            with self.assertRaises(ValueError):
                save_graph(src, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
