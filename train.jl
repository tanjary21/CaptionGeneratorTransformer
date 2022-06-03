using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random, CUDA, IterTools, DelimitedFiles, Statistics
Knet.atype() = KnetArray{Float32}
using Images, TestImages, OffsetArrays, Colors
using Plots

include("models/transformer.jl")
include("dataloader/dataloader.jl")

#------------------------------------------------------------------
#------------------------------------------------------------------
function main(args)
    # read  files
    annotations = readdlm("archive/annotations.txt", '\t', String, '\n')[2:end,2:end]

    # prepare vocab object
    bow = ""
    for sent in annotations[:,2]
        bow *= sent
    end
    #bow = split(bow)
    # bow wil be one giant sentence, corpus.
    v = Vocab(bow)

    # Initialize Iterators
    img_iterator = ImgReader("archive/img_dirs.txt",load_and_process)
    tgt_iterator = TextReader("archive/ann_caps.txt",v)

    # build Image Caption Generating Transformer
    transformer = Transformer(128, length(tgt_iterator.vocab.i2w), tgt_iterator.vocab.eos)

    # MAIN TRAINING SCRIPT  - 13 minutes/epoch
    # Dataset has 8000 images
    # Each batch is 4 images (4 unqiue, each with 5 captions, so batch size is actually 20)
    # So there will be 1:2000 iterations for one epoch
    losses = []
    for epoch in 1:1:2
        println("epoch: ", epoch)
        iter = 1
        img_state, tgt_state = nothing, nothing
        img_iterator = ImgReader("archive/img_dirs.txt",load_and_process)
        tgt_iterator = TextReader("archive/ann_caps.txt",v)
        while true #iter < 100 #flag !== nothing
            batch = get_next_batch(img_iterator, tgt_iterator, img_state, tgt_state)
            if batch == nothing
                break
            end
            batch_imgs, batch_indices, img_state, tgt_state = batch
            batch_imgs, batch_indices, labels = prepare_batch(batch, tgt_iterator)

            loss = @diff transformer(batch_imgs, batch_indices, labels)

            # updates
            for p in params(transformer)
                diff_p = grad(loss, p)
                if diff_p == nothing
                    continue
                else
                    p .= p - (0.1 .* diff_p)
                end
            end

            println("iter: ", 4*iter, "/8000", "loss: ", value(loss))
            push!(losses, value(loss))
            iter = iter + 1
        end
    end

    # plot training loss
    plot(losses)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)