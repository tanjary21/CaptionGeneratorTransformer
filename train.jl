using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random, CUDA, IterTools, DelimitedFiles, Statistics
Knet.atype() = KnetArray{Float32}
using Images, TestImages, OffsetArrays, Colors
using Plots
using JLD2

include("models/transformer.jl")
include("dataloader/dataloader.jl")

#------------------------------------------------------------------
#------------------------------------------------------------------
function main(args)
    # read  files
    annotations = readdlm("archive/annotations_train.txt", '\t', String, '\n')[2:end,2:end]

    # prepare vocab object
    bow = ""
    for sent in annotations[:,2]
        bow *= sent
    end
    #bow = split(bow)
    # bow wil be one giant sentence, corpus.
    v = Vocab(bow)

    # Initialize Iterators
    img_iterator = ImgReader("archive/img_dirs_train.txt",load_and_process)
    tgt_iterator = TextReader("archive/ann_caps_train.txt",v)

    # MAIN TRAINING SCRIPT  - 13 minutes/epoch
    # img_iterator = ImgReader("archive/img_dirs.txt",load_and_process)
    # tgt_iterator = TextReader("archive/ann_caps.txt",v)

    # Dataset has 8000 images
    # Each batch is 4 images (4 unqiue, each with 5 captions, so batch size is actually 20)
    # So there will be 1:2000 iterations for one epoch

    exp_dir = "resfreeze"
    mkpath("experiments/$exp_dir/ckpts")
    mkpath("experiments/$exp_dir/losses")

    #transformer_conv = Transformer(128, length(tgt_iterator.vocab.i2w), tgt_iterator.vocab.eos, true)
    #transformer_mlp_128 = Transformer(128, length(tgt_iterator.vocab.i2w), tgt_iterator.vocab.eos, false)
    transformer_conv_512 = Transformer(512, length(tgt_iterator.vocab.i2w), tgt_iterator.vocab.eos, true)
    #init_optimizer(transformer_conv)
    #init_optimizer(transformer_mlp_128)
    init_optimizer(transformer_conv_512)

    #losses_conv = []
    #losses_mlp_128 = []
    losses_conv_512 = [] # stores the losses across entire training session, one average per iteration/minibatch
    val_losses = []     # stores the average loss across entire val dataset, one per epoch
    train_losses = []   # stores the average loss across entire train dataset, one per epoch
    iter = 1
    for epoch in 1:1:2
        # ONE TRAINING EPOCH LOOP
        train_loss = []
        println("\n"," epoch: ", epoch, "\n")

        img_state, tgt_state = nothing, nothing
        img_iterator = ImgReader("archive/img_dirs_train.txt",load_and_process)
        tgt_iterator = TextReader("archive/ann_caps_train.txt",v)
        while true #iter < 100 #flag !== nothing
            batch = get_next_batch(img_iterator, tgt_iterator, img_state, tgt_state)
            if batch == nothing
                break
            end
            batch_imgs, batch_indices, img_state, tgt_state = batch
            batch_imgs, batch_indices, labels = prepare_batch(batch, tgt_iterator)
            batch_imgs_nomalized = 2.0 .* batch_imgs .- 1.0

            #loss_conv = @diff transformer_conv(batch_imgs, batch_indices, labels, true)
            #loss_mlp_128 = @diff transformer_mlp_128(batch_imgs, batch_indices, labels, true)
            loss_conv_512 = @diff transformer_conv_512(batch_imgs_nomalized, batch_indices, labels; p=0.1, use_smooth_loss=true)

            # updates
            #warm_adam_update(loss_conv, transformer_conv, iter)
            #warm_adam_update(loss_mlp_128, transformer_mlp_128, iter)
            warm_adam_update(loss_conv_512, transformer_conv_512, iter; freeze=true)

            println("iter: ", iter%2000, "/2000 ", "loss: ", value(loss_conv_512))
            #push!(losses_conv, value(loss_conv))
            #push!(losses_mlp_128, value(loss_mlp_128))
            push!(losses_conv_512, value(loss_conv_512))
            push!(train_loss, value(loss_conv_512))
            iter = iter + 1
        end
        push!(train_losses, mean(train_loss)) # to plot training loss, but per epoch, to compare against val
        Knet.save("experiments/$exp_dir/ckpts/transformer_conv_512_epoch$epoch.jld2","transformer",transformer_conv_512)

        # ONE VALIDATION EPOCH LOOP
        val_loss = []
        img_state, tgt_state = nothing, nothing
        img_iterator = ImgReader("archive/img_dirs_val.txt",load_and_process)
        tgt_iterator = TextReader("archive/ann_caps_val.txt",v)
        while true #iter < 100 #flag !== nothing
            batch = get_next_batch(img_iterator, tgt_iterator, img_state, tgt_state)
            if batch == nothing
                break
            end
            batch_imgs, batch_indices, img_state, tgt_state = batch
            batch_imgs, batch_indices, labels = prepare_batch(batch, tgt_iterator)
            batch_imgs_nomalized = 2.0 .* batch_imgs .- 1.0

            #loss_conv = @diff transformer_conv(batch_imgs, batch_indices, labels, true)
            #loss_mlp_128 = @diff transformer_mlp_128(batch_imgs, batch_indices, labels, true)
            push!(val_loss, transformer_conv_512(batch_imgs_nomalized, batch_indices, labels; p=0.0, use_smooth_loss=true))

        end
        push!(val_losses, mean(val_loss))
    end

    save_object("experiments/$exp_dir/losses/losses_conv_512.jld2", losses_conv_512)
    save_object("experiments/$exp_dir/losses/val_losses.jld2", val_losses)
    save_object("experiments/$exp_dir/losses/train_losses.jld2", train_losses)

    # plot training loss - next cell
    #plot([losses_conv, losses_mlp_128, losses_mlp_512], labels=["conv" "mlp-128" "mlp-512"],xlabel="iterations",ylabel="NLL Loss")    
    
    
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)