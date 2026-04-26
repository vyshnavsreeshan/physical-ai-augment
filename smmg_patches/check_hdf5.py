"""Check generated_dataset.hdf5 for actual motion."""
import h5py, numpy as np, os

p = "/workspace/datasets/generated_dataset.hdf5"
print("file exists:", os.path.exists(p), "size:", os.path.getsize(p) if os.path.exists(p) else 0)
if not os.path.exists(p):
    raise SystemExit(0)
with h5py.File(p, "r") as f:
    demos = list(f["data"].keys())
    print(f"demos: {demos}")
    if not demos:
        raise SystemExit(0)
    d = f["data"][demos[0]]
    print(f"top-level keys: {list(d.keys())}")
    print(f"attrs: {dict(d.attrs)}")
    if "actions" in d:
        a = d["actions"][:]
        print(f"actions shape: {a.shape}  range=[{a.min():.3f}, {a.max():.3f}]  per-step std mean={a.std(axis=0).mean():.4f}")
        print(f"  action[0] = {a[0]}")
        print(f"  action[-1] = {a[-1]}")
    if "obs" in d:
        print(f"obs keys: {list(d['obs'].keys())}")
        for k in list(d["obs"].keys())[:5]:
            arr = d["obs"][k]
            if arr.ndim >= 2 and arr.shape[0] > 1:
                diff = np.abs(arr[0].astype(np.float32) - arr[-1].astype(np.float32)).mean()
                print(f"  {k}: shape {arr.shape}  first↔last diff={diff:.4f}")
    if "states" in d:
        s = d["states"]
        if isinstance(s, h5py.Group):
            print(f"states keys: {list(s.keys())}")
            for k in list(s.keys())[:3]:
                arr = s[k]
                if hasattr(arr, "shape") and arr.ndim >= 2 and arr.shape[0] > 1:
                    diff = np.abs(arr[0].astype(np.float32) - arr[-1].astype(np.float32)).mean()
                    print(f"  states/{k}: shape {arr.shape}  diff={diff:.4f}")
