import torch

def save_state():
    obj = torch.load('saved_runs/2025-10-31_21-20-46_student_mbv3s_vww96_kd_refine/model.pt', map_location="cpu", weights_only=False)
    pruned = obj["model"]
    torch.save(pruned.state_dict(), "saved_runs/2025-10-31_21-20-46_student_mbv3s_vww96_kd_refine/model_state.pt")

def dump_dtypes(path: str):
    sd = torch.load(path, map_location="cpu")
    state = sd.get("model_state", sd.get("model", sd))
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            print(k, v.dtype)

def main():
    save_state()
    #dump_dtypes("saved_runs/2025-10-31_21-20-46_student_mbv3s_vww96_kd_refine/model.pt")
    #dump_dtypes("runs/2025-11-15_16-05-11_student_mbv3s_vww96_quant/model_int8_ptq_state.pt")


if __name__ == "__main__":
    main()
