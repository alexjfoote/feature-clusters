def to_dict(mapping):
    mapping = {
        tuple(int(n) for n in k.split("_")) if "_" in k else int(k): v for k, v in mapping.items()
    }
    return mapping


def to_json(mapping):
    mapping = {"_".join([str(e) for e in k]) if isinstance(k, tuple) else k: v for k, v in mapping.items()}
    return mapping



def get_by_id(_id, id_map, arr):
    idxs = id_map.get(_id, [])
    return [arr[idx] for idx in idxs]

