import importlib

def call_load_dataset(cfg):
    name = cfg.dataset
    print("name:", name)

    key = name.split("-")[0]
    module_name = f"datasets.{key}"
    function_name = "load_datasets"

    if cfg.visual:
        function_name = function_name + "_" + "visual"

    if cfg.prompt == "coarse":
        function_name = function_name + "_" + "coarse"

    module = importlib.import_module(module_name)
    func = getattr(module, function_name)
    return func
