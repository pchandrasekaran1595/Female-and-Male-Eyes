import sys

import utils as u
import Models
from build_dataloaders import build
from api import fit, predict

#####################################################################################################

def app():
    args_1 = "--bs"
    args_2 = "--lr"
    args_3 = "--wd"
    args_4 = "--scheduler"
    args_5 = "--epochs"
    args_6 = "--early"
    args_7 = "--train-full"
    args_8 = "--test"
    args_9 = "--name"
    
    train_mode = True
    train_full = None
    do_scheduler = None
    scheduler = None
    batch_size, lr, wd = 64, 1e-3, 0
    epochs = 10
    early_stopping = 5
    name = "Test_1.jpg"

    if args_1 in sys.argv:
        batch_size = int(sys.argv[sys.argv.index(args_1) + 1])
    if args_2 in sys.argv:
        lr = float(sys.argv[sys.argv.index(args_2) + 1])
    if args_3 in sys.argv:
        wd = float(sys.argv[sys.argv.index(args_3) + 1])
    if args_4 in sys.argv:
        do_scheduler = True
        patience = int(sys.argv[sys.argv.index(args_4) + 1])
        eps = float(sys.argv[sys.argv.index(args_4) + 2])
    if args_5 in sys.argv:
        epochs = int(sys.argv[sys.argv.index(args_5) + 1])
    if args_6 in sys.argv:
        early_stopping = int(sys.argv[sys.argv.index(args_6) + 1])
    if args_7 in sys.argv:
        train_full = True
    if args_8 in sys.argv:
        train_mode = False
    if args_9 in sys.argv:
        name = sys.argv[sys.argv.index(args_9) + 1]

    if train_mode:
        dataloaders = build(path="./", batch_size=batch_size)

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
