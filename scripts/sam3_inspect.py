"""Print how a SAM 3 decoder graph turns its raw outputs into ``scores``: the nodes behind each output, with small
constants (e.g. a confidence cut baked in by an export that holds Sam3Processor's post-processing).

    python scripts/sam3_inspect.py path/to/sam3-decoder.onnx [--depth 14]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> None:
    import onnx
    from onnx import numpy_helper

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('graph')
    ap.add_argument('--depth', type=int, default=14)
    a = ap.parse_args()
    m = onnx.load(a.graph, load_external_data=False)
    g = m.graph
    const = {}
    for t in g.initializer:
        if not t.external_data and np.prod(t.dims or [1]) <= 4:
            const[t.name] = numpy_helper.to_array(t).ravel().tolist()
    producer = {}
    for n in g.node:
        for o in n.output:
            producer[o] = n
        if n.op_type == 'Constant':
            for at in n.attribute:
                if at.name == 'value' and np.prod(at.t.dims or [1]) <= 4:
                    const[n.output[0]] = numpy_helper.to_array(at.t).ravel().tolist()
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    print('outputs:', [o.name for o in g.output])
    print('Greater/Less/GreaterOrEqual nodes with a constant side:')
    for n in g.node:
        if n.op_type in ('Greater', 'Less', 'GreaterOrEqual', 'LessOrEqual'):
            c = [const.get(i) for i in n.input]
            if any(v is not None for v in c):
                cons = [x.op_type for x in consumers.get(n.output[0], [])]
                src = [producer[i].op_type if i in producer else 'input/init' for i in n.input]
                print(f'  {n.name} {n.op_type} inputs={list(n.input)} from={src} const={c} -> {cons}')
    for out in ('scores', 'boxes'):
        print(f'behind {out}:')
        seen, frontier = set(), [(out, 0)]
        while frontier:
            name, d = frontier.pop(0)
            n = producer.get(name)
            if n is None or n.name in seen or d > a.depth:
                continue
            seen.add(n.name)
            c = {i: const[i] for i in n.input if i in const}
            print(f'  {"  " * d}{n.op_type} {n.name} in={list(n.input)}{f" const={c}" if c else ""}')
            frontier += [(i, d + 1) for i in n.input if i not in const]
    sys.stdout.flush()


if __name__ == '__main__':
    main()
