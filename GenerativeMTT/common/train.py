# Common training-related configs that are designed for "tools/lazyconfig_train_net_kfold_test.py"
# You can use your own instead, together with your own train_net.py
train = dict(
    output_dir="",
    init_checkpoint="",
    max_iter=90000,
    amp=dict(enabled=False,
             grad_clipper=None),  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    checkpointer=dict(period=5000,
                      max_to_keep=100),  # options for PeriodicCheckpointer
    eval_period=5000,
    split_combine=dict(
        enabled=False,
        split_combiner=None,
        inference_batch_size=1,
    ),
    log_period=20,
    device="cuda"
    # ...
)
