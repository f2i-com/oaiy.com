"""Print YuNet's ONNX graph: inputs, outputs, each node with its attributes, and the initializers' shapes."""
import sys
from collections import Counter

import onnx
from onnx import numpy_helper

m = onnx.load(sys.argv[1])
g = m.graph
print("opset", [(o.domain, o.version) for o in m.opset_import])
for i in g.input:
    print("input", i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
for o in g.output:
    print("output", o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
print("initializers", len(inits), "params", sum(a.size for a in inits.values()))
print("ops", Counter(n.op_type for n in g.node))
for n in g.node:
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
    attrs = {k: (v if not isinstance(v, bytes) else v.decode()) for k, v in attrs.items()}
    ins = [f"{x}{list(inits[x].shape) if x in inits else ''}" for x in n.input]
    print(n.op_type, n.name, ins, "->", list(n.output), attrs)
