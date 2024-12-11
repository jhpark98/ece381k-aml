class CFG:
    DEMO_MODE = False
    DEVICE = 'cuda'
    DATA_DIR = "/path/to/data/"
    TRAIN_CSV = "/path/to/train_events.csv"
    TEST_CSV = "/path/to/test_series.parquet"
    # Model parameters
    ARCH = [(2, 8, 2, 17), 
            (8, 32, 2, 11), 
            (32, 64, 2, 7)]
    KERNEL_SIZE = 25
    STRIDE = 2
    N_LAYERS = 5
    SIGMA = 36
    MAX_CHUNK_SIZE = 24*60*12
    LEN_MULT = 2**len(ARCH)