import sys
import torch

import utils as u
import Models
from build_dataloaders import build
from api import fit, predict

#####################################################################################################

def app():
    args_0 = "--full"
    args_1 = "--bs"
    args_2 = "--lr"
    args_3 = "--wd"
    args_4 = "--scheduler"
    args_5 = "--epochs"
    args_6 = "--early"
    args_7 = "--train-full"
    args_8 = "--augment"
    args_9 = "--test"
    args_10 = "--name"
    
    do_full = None
    train_mode = True
    train_full = None
    do_scheduler = None
    scheduler = None
    do_augment = None
    batch_size, lr, wd = 64, 1e-3, 0
    epochs = 10
    early_stopping = 5
    name = "Test_1.jpg"

    if args_0 in sys.argv: do_full = True
    if args_1 in sys.argv: batch_size = int(sys.argv[sys.argv.index(args_1) + 1])
    if args_2 in sys.argv: lr = float(sys.argv[sys.argv.index(args_2) + 1])
    if args_3 in sys.argv: wd = float(sys.argv[sys.argv.index(args_3) + 1])
    if args_4 in sys.argv:
        do_scheduler = True
        patience = int(sys.argv[sys.argv.index(args_4) + 1])
        eps = float(sys.argv[sys.argv.index(args_4) + 2])
    if args_5 in sys.argv: epochs = int(sys.argv[sys.argv.index(args_5) + 1])
    if args_6 in sys.argv: early_stopping = int(sys.argv[sys.argv.index(args_6) + 1])
    if args_7 in sys.argv: train_full = True
    if args_8 in sys.argv: do_augment = True
    if args_9 in sys.argv: train_mode = False
    if args_10 in sys.argv: name = sys.argv[sys.argv.index(args_10) + 1]

    if train_mode:
        dataloaders = build(path="./", batch_size=batch_size, do_full=do_full, do_augment=do_augment)

        torch.manual_seed(u.SEED)
        model = Models.ResNet50(train_full=train_full)
        optimizer = model.getOptimizer(lr=lr, wd=wd)
        if do_scheduler:
            scheduler = model.getPlateauScheduler(optimizer=optimizer, patience=patience, eps=eps)

        L, A, _, _ = fit(model=model, optimizer=optimizer, scheduler=scheduler, epochs=epochs,
                        early_stopping_patience=early_stopping, dataloaders=dataloaders, verbose=True)
        u.save_graphs(L, A)
    else:
        assert(isinstance(name, str))

        image = u.read_image(name)
        
        model = Models.ResNet50(train_full=True)
        u.breaker()
        print("{} : {}".format(name, predict(model, image)))
        u.breaker()

#####################################################################################################
