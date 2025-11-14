import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from pathlib import Path

from Module.Frontend.Frontend import IFrontend#, ModelContext
from Module.Frontend.Matching import IMatcher
from Module.Frontend.StereoDepth import IStereoDepth
from Utility.Config import load_config


# def GetFlops(model: IFrontend[ModelContext] | IMatcher[ModelContext] | IStereoDepth[ModelContext]):
def GetFlops(model):
    input_A = torch.rand(2, 3, 640, 640, device='cpu')
    input_B = torch.rand(2, 3, 640, 640, device='cpu')
    print(type(model.model))
    flops = FlopCountAnalysis(model.model, (input_A, input_B))
    print(f"Total Flops: {flops.total()/1e9} G")
    print(flop_count_table(flops))


def main(cfg_path: str, module_type: str):
    cfg, _ = load_config(Path(cfg_path))
    
    # module: IMatcher[ModelContext] | IStereoDepth[ModelContext] | IFrontend[ModelContext]
    match module_type:
        case "frontend":
            module = IFrontend.instantiate(cfg.Odometry.frontend.type, cfg.Odometry.frontend.args)
        case "depth":
            module = IStereoDepth.instantiate(cfg.depth.type, cfg.args)
        case "flow":
            module = IMatcher.instantiate(cfg.flow.type, cfg.args)
        case _:
            raise ValueError(f"module_type Must be one of 'frontend', 'depth' or 'flow'. Get {_}")
    
    GetFlops(module)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--module", choices=["frontend", "depth", "flow"])
    args = parser.parse_args()
    
    main(args.config, args.module)

