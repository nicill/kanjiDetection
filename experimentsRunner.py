class ExperimentRunner:

    def __init__(self, conf):
        self.conf = conf
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Preparing datasets once...")
        self.base = ModelExperiment(conf, self.device)
        self.base.prepare_data()
        self.yolo = YOLOExperiment(conf, self.device)
        self.pytorch = PyTorchModelExperiment(conf, self.device)
        self.detr = DETRExperiment(conf, self.device)

    def run_grid(self, experiment, grid, name):
        print(f"\n=== {name} GRID ===")
        for params in grid:
            print(f"\n@@@@ {name} {params}")
            try:
                metrics, train_time, test_time = experiment.train_and_test(params)
                experiment.write_results(params, metrics, train_time, test_time)
                print(f"Prec:{metrics['oprec']:.4f} Rec:{metrics['orec']:.4f}")
            except Exception as e:
                print(f"{name} failed: {e}")
            torch.cuda.empty_cache(); gc.collect()