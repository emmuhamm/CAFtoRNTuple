import sys
from collections import defaultdict
import uproot

def load(path, name="caf"):
    nt = uproot.open(path)[name]
    recs = nt.header.field_records          # flat list of all fields
    kids = defaultdict(list)
    for i, f in enumerate(recs):
        # top-level fields have parent_field_id == their own id
        kids[None if f.parent_field_id == i else f.parent_field_id].append(i)
    return nt, recs, kids

def walk(recs, kids, parent, depth, max_depth, path, flat, hide_elem):
    for i in kids[parent]:
        f = recs[i]
        name = f.field_name
        is_elem = hide_elem and name == "_0"   # vector element wrapper
        full = path if is_elem else (f"{path}.{name}" if path else name)
        if not is_elem:
            if flat:
                print(f"{full}   [{f.type_name}]")
            else:
                print("  " * depth + f"{name}  [{f.type_name}]")
        if depth + 1 < max_depth:
            walk(recs, kids, i, depth + 1, max_depth, full, flat,
                 hide_elem)

def find(recs, kids, dotted):
    """Return field id for a dotted path like 'rec.nd.lar'."""
    parent = None
    for part in dotted.split("."):
        parent = next(i for i in kids[parent] if recs[i].field_name == part)
    return parent

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("-n", "--name", default="caf")
    ap.add_argument("-d", "--depth", type=int, default=3)
    ap.add_argument("-s", "--start", default="")
    ap.add_argument("--flat", action="store_true", help="one dotted path per line")
    ap.add_argument("--keep-elem", action="store_true", help="show _0 element fields")
    a = ap.parse_args()

    nt, recs, kids = load(a.file, a.name)
    start = find(recs, kids, a.start) if a.start else None
    walk(recs, kids, start, 0, a.depth, a.start, a.flat, not a.keep_elem)