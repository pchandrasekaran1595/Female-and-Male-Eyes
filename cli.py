import sys

import utils as u
import Models
from build_dataloaders import build
from api import fit, predict

def app():
    args_1 = "--colab"
    args_2 = "--bs"
    args_3 = "--lr"
    args_4 = "--wd"
    args_5 = "--scheduler"
    args_6 = "--epochs"
    args_7 = "--early"
    args_8 = "--train-full"
    args_9 = "--test"
    
    train_mode = True
    train_full = None
    in_colab = None
    do_scheduler = None
    scheduler = None
    batch_size, lr, wd = 64, 1e-3, 0
    epochs = 10
    early_stopping = 5

    if args_1 in sys.argv:
        in_colab = True
    if args_2 in sys.argv:
        batch_size = int(sys.argv[sys.argv.index(args_2) + 1])
    if args_3 in sys.argv:
        lr = float(sys.argv[sys.argv.index(args_3) + 1])
    if args_4 in sys.argv:
        wd = float(sys.argv[sys.argv.index(args_4) + 1])
    if args_5 in sys.argv:
        do_scheduler = True
        patience = int(sys.argv[sys.argv.index(args_5) + 1])
        eps = float(sys.argv[sys.argv.index(args_5) + 2])
    if args_6 in sys.argv:
        epochs = int(sys.argv[sys.argv.index(args_6) + 1])
    if args_7 in sys.argv:
        early_stopping = int(sys.argv[sys.argv.index(args_7) + 1])
    if args_8 in sys.argv:
        train_full = True
    if args_9 in sys.argv:
        train_mode = False
    
    if train_mode:
        if in_colab:
            dataloaders = build(path="/content", batch_size=batch_size, in_colab=in_colab)
        else:
            dataloaders = build(path="./", batch_size=batch_size, in_colab=in_colab)

        model = Models.ResNet50(train_full=train_full)
        optimizer = model.getOptimizer(lr=lr, wd=wd)
        if do_scheduler:
            scheduler = model.getPlateauScheduler(optimizer=optimizer, patience=patience, eps=eps)

        L, A, _, _ = fit(model=model, optimizer=optimizer, scheduler=scheduler, epochs=epochs,
                        early_stopping_patience=early_stopping, dataloaders=dataloaders, verbose=True)
        
        u.save_graphs(L, A)
    else:
        raise NotImplementedError("Testing not Implemented as yet")
    
    

    
