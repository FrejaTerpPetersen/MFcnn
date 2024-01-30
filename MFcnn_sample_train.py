import torch
def train_MFcnn(LFmod,HFmod,num_train,num_val,cnn,nT_in,lr,dt,nT=52,n_epochs=1000):
    # Inputs: 
    #       LFmod:          A function which outputs the low-fidelity data, given an input of the number of training data and the number of time steps.
    #                       The function makes sure that there is variation in data. 
    #                       Output is a torch.tensor with shape (num_train,nT,1,Nx)
    #       HFmod:          A high-fidelity function similar to above.
    #       num_train:      Number of training data.
    #       num_val:        Mumber of validation data.
    #       cnn:            A PyTorch formulation of a CNN which takes input U where U contains the state variables. 
    #                       U is a torch.tensor with shape (1,nT_in,Nx), where nT_in is the number of previous time steps included and Nx
    #                       is the shape in the x-direction
    #       nT_in:          Number of previous time steps that the CNN takes as input.
    #       lr:             Initial learning rate of optimizer
    #       dt:             Time step size
    #       nT:             Number of time steps during training
    #       n_epochs:       Number of epochs for the training loop, ie. number of times iterating through training data.

    # Prepare data
    # Data has dimension: (num_train,nT,1,Nx)
    HF_train = HFmod(num_train,nT)
    LF_train = LFmod(num_train,nT)
    # Data has dimension: (num_val,nT,1,Nx)
    HF_val = HFmod(num_val,nT)
    LF_val = LFmod(num_val,nT)
    Nx = LF_val.shape[3]

    K = cnn.kernel_size # Kernel size of CNN
    # number of boundary values to be explicitly overwritten during training
    Nb = int(torch.floor(torch.tensor(K/2)))
    # Set up optimizer
    optimizer = torch.optim.Adam(cnn.parameters(),lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1,patience=30,min_lr=1e-8)
    loss_fn = torch.nn.MSELoss()
    min_loss = 1e-5
    train_best = 1e-5

    # Time steps to loop over. 
    nT_loop = nT-1-nT_in

    for epoch in range(n_epochs):
        mean_loss = 0
        for t in range(nT_loop):
            mean_loss_time = 0
            for i in range(num_train):

                # Next time step with LF-model
                LF_out = LF_train[i,t+nT_in]

                # Next time step with MF-model
                cnn_in = HF_train[i,t:(t+nT_in)].reshape(1,nT_in,Nx) # Use HF-data as input to CNN
                MF_out = LF_out + dt*cnn(cnn_in) # step with MF-model

                # Next time step with HF-model for reference
                HF_out = HF_train[i,t+nT_in]

                # Explicitly correct boundaries using reference solution
                MF_out[:,:,0:Nb] = HF_out[:,:,0:Nb]
                MF_out[:,:,Nx-Nb:] = HF_out[:,:,Nx-Nb:]

                loss = loss_fn(HF_out,MF_out)           # Calculate loss
                optimizer.zero_grad()                   # Reset gradients
                loss.backward()                         # Compute gradients
                optimizer.step()                        # Step with optimizer

                mean_loss_time += loss/(num_train)
            mean_loss += mean_loss_time/nT

        # Update learning rate with scheduler.
        lr_bef =scheduler.optimizer.param_groups[0]['lr']
        scheduler.step(mean_loss)
        lr_aft =scheduler.optimizer.param_groups[0]['lr']
        if lr_bef!= lr_aft:
            print("lr changed from ",lr_bef, " to ", lr_aft)

        # Print loss
        if epoch % 5 == 0:
            print("[epoch]", epoch, "Train mean loss:", mean_loss.item())

        if  mean_loss < train_best:
            train_best = mean_loss
            with torch.no_grad():
                mean_valloss = 0
                mean_valmse = 0
                for i in range(num_val):
                    for t in range(nT_loop):
                        # Next time step with LF-model
                        LF_out = LF_val[i,t+nT_in]

                        # Next time step with MF-model
                        cnn_in = HF_val[i,t:(t+nT_in)].reshape(1,nT_in,Nx) # Input to CNN
                        MF_out = LF_out + dt*cnn(cnn_in)

                        # Next time step with HF-model for reference
                        HF_out = HF_val[i,t+nT_in]

                        # Explicitly correct boundaries using reference solution
                        MF_out[:,:,0:Nb] = HF_out[:,:,0:Nb]
                        MF_out[:,:,Nx-Nb:] = HF_out[:,:,Nx-Nb:]

                        loss = loss_fn(HF_out,MF_out)    # Calculate loss
                        optimizer.zero_grad()                   # Reset gradients
                        loss.backward()                         # Compute gradients
                        optimizer.step()                        # Step with optimizer

                        # Take loss in last time step
                        if t == nT_loop -1:
                            val_mse = loss

                    mean_valmse += val_mse/num_val
                if mean_valmse < min_loss:
                    # Save model, if it is the best one so far. 
                    torch.save(cnn.state_dict(),
                            'MF_cnn.pt')
                    min_loss = mean_valloss
                    print("[epoch]", epoch, "Train Loss: ", train_best.item(), "Mean relative L2 validation error: ", mean_valloss)