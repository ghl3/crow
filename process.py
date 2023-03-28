from collections import defaultdict
import pandas as pd


def state_list_to_frame_dict(physics, states):
    body_names = [
        physics.model.id2name(body_id, "body") for body_id in range(physics.model.nbody)
    ][1:]

    data = defaultdict(list)
    for state in states:
        df = pd.DataFrame(
            state["pos"],
            columns=["x", "y", "z"],
            index=body_names,
        )
        for ix, row in df.iterrows():
            d = row.to_dict()
            d["time"] = state["time"]
            data[ix].append(d)

    data = {
        k: pd.DataFrame(rows).set_index("time").sort_index() for k, rows in data.items()
    }
    return data
