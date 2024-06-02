import os
import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._functorch.aot_autograd import aot_autograd_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg


def foo(x, y):
    return x + y

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=linalg.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)
in1 = torch.randn(5, dtype=torch.float32)
in2 = torch.randn(5, dtype=torch.float32)
graphs = dynamo_compiler.importer(foo, in1, in2)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
path_prefix = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(path_prefix, "f16add.mlir"), "w") as module_file:
    print(graph._imported_module, file=module_file)