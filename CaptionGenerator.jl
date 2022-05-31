using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random, CUDA, IterTools, DelimitedFiles, Statistics
Knet.atype() = KnetArray{Float32}
using Images, TestImages, OffsetArrays, Colors
using Plots

# Data Preprocessing and Loading/Iterating

###########################################
############### VOCABULARY ################
###########################################
struct Vocab
    w2i::Dict{String,Int}
    i2w::Vector{String}
    unk::Int
    eos::Int
    tokenizer
end

function Vocab(bow::String; tokenizer=split, vocabsize=Inf, mincount=1, unk="<unk>", eos="<eos>")
    # Your code here
    #words = read(file, String)
    words = tokenizer(lowercase(bow))
    
    word_histogram = Dict(unk=>0,eos=>0)
    for word in words
        word_histogram[word] = get!(word_histogram, word, 0) + 1
    end
    
    words = collect(keys(word_histogram))
    freqs = collect(values(word_histogram))
    idxs = freqs .>= mincount
    words = words[idxs]
    freqs = freqs[idxs]
    sort_idxs = sortperm(freqs, rev=true)
    i2w = words[sort_idxs]
    freqs = freqs[sort_idxs]
    unk in i2w ? nothing : pushfirst!(i2w, unk)
    eos in i2w ? nothing : pushfirst!(i2w, eos)
    i2w = i2w[1:(vocabsize>length(i2w) ? end : vocabsize)]
    w2i = Dict(zip(i2w, 1:length(i2w)))
    Vocab(w2i, i2w, w2i[unk], w2i[eos], tokenizer)
end

###########################################
############### TEXT READER ###############
###########################################
struct TextReader
    file::String
    vocab::Vocab
end

function Base.iterate(r::TextReader, s=nothing)
    # Your code here
    if s == nothing
        s = open(r.file)
        header = readline(s) # to discard header
    end
    if eof(s)
        close(s)
        return nothing
    else
        sentence = readline(s)
        words = r.vocab.tokenizer(sentence)
        indices = []
        for word in words
            push!(indices, get(r.vocab.w2i, word, r.vocab.unk))
        end
#         indices = collect(getindex.(Ref(r.vocab.w2i), tuple(words...)))
    end
    return Vector{Int64}(indices), s
end

Base.IteratorSize(::Type{TextReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TextReader}) = Base.HasEltype()
Base.eltype(::Type{TextReader}) = Vector{Int}

###########################################
############### IMAGE READER ##############
###########################################
struct ImgReader
    file::String
    preprocessor
    
end

function Base.iterate(r::ImgReader, s=nothing)
    # Your code here
    if s == nothing
        s = open(r.file)
        header = readline(s) # to discard header 
    end
    if eof(s)
        close(s)
        return nothing
    else
        img_name = split(readline(s))[2]
        img_tensor = r.preprocessor("archive/Images/$img_name")
        
    end
    return img_tensor, s
end

Base.IteratorSize(::Type{ImgReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{ImgReader}) = Base.HasEltype()
Base.eltype(::Type{ImgReader}) = Vector{Int}

###########################################
############# HELPER FUNCTION #############
###########################################
function load_and_process(img_path::String)
    rgb_img = load(img_path) # "archive/Images/1000268201_693b08cb0e.jpg"
    rgb_img = imresize(rgb_img, (400, 400))
    img = KnetArray{Float32}(channelview(rgb_img))
end


###########################################
############### DATALOADING ###############
###########################################
function get_next_batch(img_iterator, tgt_iterator, img_state=nothing, tgt_state=nothing)
    
    batch_indices = []
    batch_imgs = []
    for i in 1:20
        tgt = iterate(tgt_iterator, tgt_state)
        img = iterate(img_iterator, img_state)
        if tgt == nothing || img == nothing
            return nothing
        end
        indices, tgt_state = tgt
        img_tensor, img_state = img
        
        # img_tensor is 3x400x400 Knet array from the iterator
        img_tensor = permutedims(img_tensor, (2,3,1)) # make channel dim at the end for resnent convs
        img_tensor = reshape(img_tensor,(size(img_tensor)...,1)) # add signleton batch dimesion
        push!(indices, tgt_iterator.vocab.eos)
        pushfirst!(indices, tgt_iterator.vocab.eos)
        push!(batch_imgs, img_tensor)
        push!(batch_indices, indices)
    end
    #return cat(batch_indices...; dims=2), tgt_state
    return batch_imgs, batch_indices, img_state, tgt_state
end  
function mask!(a,pad)
    # Your code here
    a = reverse(a, dims = 2)
    ind = [findfirst(a[i, :].!=pad) for i in 1:size(a,1)]
    for i in 1:size(a,1)
        if ind[i] > 2
            a[i, 1:ind[i]-2] .= 0
        end
    end
    a = reverse(a, dims = 2)
end

function prepare_batch(batch, tgt_iterator)
    batch_imgs, batch_indices, img_state, tgt_state = batch
    longest = maximum(length.(batch_indices))
    batch_size = length(batch_indices)
    square_batch_indices = tgt_iterator.vocab.eos .* ones(Int64, batch_size, longest)
    for (i, sent) in enumerate(batch_indices)
        square_batch_indices[i,1:length(sent)] = sent
    end
    batch_imgs = cat(batch_imgs...;dims=4)
    batch_indices = square_batch_indices[:,1:end-1]
    labels = mask!(square_batch_indices[:,2:end],tgt_iterator.vocab.eos)
    return batch_imgs, Matrix{Int64}(transpose(batch_indices)), Matrix{Int64}(transpose(labels))
end

# MODEL DEFINITIONS
include("resnet/resnet.jl")
include("resnet/resnetlib.jl")

# Some base modules
###########################################
################# EMBED ###################
###########################################
struct Embed; w; end

function Embed(vocabsize::Int, embedsize::Int)
    # Your code here
    Embed(param(embedsize,vocabsize))
end

function (l::Embed)(x)
    # Your code here
    l.w[:,x]
end

###########################################
################# LINEAR ##################
###########################################
struct Linear; w; b; end

function Linear(inputsize::Int, outputsize::Int)
    Linear(param(outputsize,inputsize),param0(outputsize))
end

function (l::Linear)(x)
    #l.w*x .+ l.b
    if length(size(x)) < 3
        result = l.w * mat(x, dims = 1) .+ l.b
    else
        in_feats, num_t, b = size(x)
        out_feats = size(l.w, 1)
        result = l.w * reshape(x, (in_feats, num_t*b)) .+ l.b
        result = reshape(result, (out_feats, num_t, b))
    end
    result
end

###########################################
################ PRE-CONV #################
###########################################
struct Preconv; w; end
function Preconv()
    w = Any[xavier(Float32, 7,7,256,20), zeros(Float32,1,1,1,1,1),
            xavier(Float32,7,7,20,1), zeros(Float32,1)]
    w = map(Knet.array_type[], w)
    w = map(param, w)
    Preconv(w)
end
function (c::Preconv)(x)
    #temp = pool(conv4(c.w[1], x))# .+ w[2]))
    #a = pool(conv4(c.w[3], temp))# .+ w[4]))
    temp = pool(relu.(conv4(c.w[1], x)))# .+ w[2]))
    a = pool(relu.(conv4(c.w[3], temp)))# .+ w[4]))
end

###########################################
################# RESNET ##################
###########################################
struct MyResNet; w; m; meta; end
function MyResNet()
    w, m, meta = ResNetLib.resnet50init(trained=true)#, etype=KnetArray{Float32})#, stage=0)
    w = map(Knet.array_type[], w)
    w = map(param, w)
    MyResNet(w, m, meta)
end
function (r::MyResNet)(x)
    resnet_feats = ResNetLib.resnet50(r.w, r.m, x; stage=1) # 99x99x256*B
end

###########################################
################ SELF ATTN ################
###########################################
#function self_attn(q::KnetArray{Float32, 3}, k::KnetArray{Float32, 3}, v::KnetArray{Float32, 3}; mask=false)
function self_attn(q, k, v; mask=false)
    a = bmm(permutedims(k,(2,1,3)),q) # num_kv x num_q x B
    if mask
        mask_tensor = Knet.atype(UpperTriangular(ones(size(a,2),size(a,2)))) # upper half including diag will be ones
        # replace zeros with large negative number before softmax
        mask_tensor = (-1e12.* (1.0 .- mask_tensor)) # lower half excluding diag will be large negative, upper half will be zeros
        a = a .+ mask_tensor
    end
    
    
    a = softmax(a; dims=1)./Float32(sqrt(128))
    updated_q = bmm(v,a) # 128 x num_q x B
end

###########################################
############### CROSS ATTN ################
###########################################
#function cross_attn(q::Knet.atype(), k::Knet.atype(), v::Knet.atype())
function cross_attn(q, k, v)
    a = bmm(permutedims(k,(2,1,3)),q) # num_kv x num_q x B
    a = softmax(a; dims=1)./Float32(sqrt(128))
    updated_q = bmm(v,a) # 128 x num_q x B
end

###########################################
############### LAYER NORM ################
###########################################
struct LayerNorm; a; b; end
function LayerNorm(token_size::Int)
    LayerNorm(param(128,1), param0(128,1))
end
#function (ln::LayerNorm)(x::Knet.atype())
function (ln::LayerNorm)(x)
    avgs = mean(x, dims=1)
    stds = std(x, dims=1)
    ln.a .* (x .- avgs) ./ (stds .+ 1e-12) .+ ln.b
end




# some composite modules
###########################################
############## IMAGE ENCODER ##############
###########################################
struct ImageEncoder; token_size; resnet; preconv; precoder; end
function ImageEncoder(token_size::Int)
    resnet = MyResNet() # 99x99x256*B
    preconv = Preconv()
    precoder = Linear(20*20*1,1024)
    ImageEncoder(token_size, resnet, preconv, precoder)
end
#function (i::ImageEncoder)(x::KnetArray{Float32, 4})
function (i::ImageEncoder)(x)
    # takes a batch of images 400x400x3xB
    # return tokens token_size x num_t x B
    resnet_feats = i.resnet(x) # 99x99x256*B
    precoder_output = i.preconv(resnet_feats) # 20x20x1xB
    h, w, c, b = size(precoder_output)
    temp = reshape(precoder_output, (h*w*c,b)) # 20*20*1xB
    o = i.precoder(temp) # 512xB (all tokens in a single vector x B)
    F, B = size(o) # F(whole feature vector size 512) B(batch size)
    img_tokens = reshape(o,(i.token_size, div(F,i.token_size), B)) # 128 x num_q x B
    img_tokens .+ oneD_PE(img_tokens)
end

###########################################
######## 1D POSITIONAL EMBEDDINGS #########
###########################################
function oneD_PE(word_tokens)#::KnetArray{Float32, 3})
    pe = zeros(Float32, size(word_tokens)...)
    pe = Knet.atype(pe)
    token_size, num_t, B = size(word_tokens)
    w = (0:2:token_size-1).*ones(div(token_size,2))#(1:2:128)
    w = 1 ./ (1000 .^ w)
    w = repeat(reshape(w, (div(token_size,2), 1)),1, num_t, B)

    pi = permutedims(repeat((0:1:num_t-1).*ones(num_t,1), 1, token_size, B),(2,1,3))

    pe[1:2:128,:,:] = sin.(pi[1:2:128,:,:].*w)
    pe[2:2:128,:,:] = cos.(pi[2:2:128,:,:].*w)
    pe = pe .+ 1e-12
    return pe
end

###########################################
############ SENTENCE ENCODER #############
###########################################
struct SentEncoder; embed::Embed; end
function SentEncoder(token_size=128, vocab_size=10000)
    SentEncoder(Embed(vocab_size, token_size))
end
#function (s::SentEncoder)(x::Matrix{Int64})
function (s::SentEncoder)(x)
    # takes a batch of worindices as input num_t x B
    # returns embeddings token_size x num_t x B
    word_tokens = s.embed(x)
    word_tokens .+ oneD_PE(word_tokens)
end

###########################################
############## ENCODER LAYER ##############
###########################################
struct EncoderLayer
    n1::LayerNorm
    q_mlp::Linear
    k_mlp::Linear
    v_mlp::Linear
    
    n2::LayerNorm
    ffn::Linear
end
function EncoderLayer(token_size::Int)
    EncoderLayer(LayerNorm(token_size),
            Linear(token_size, token_size),
            Linear(token_size, token_size),
            Linear(token_size, token_size),
            LayerNorm(token_size),
            Linear(token_size, token_size))
end
#function (el::EncoderLayer)(q::KnetArray{Float32, 3}, k::KnetArray{Float32, 3}, v::KnetArray{Float32, 3})
function (el::EncoderLayer)(q, k, v)
    # self attn
    q_ = relu.(el.q_mlp(el.n1(q)))
    k_ = relu.(el.k_mlp(el.n1(k)))
    v_ = relu.(el.v_mlp(el.n1(v)))
    q = self_attn(q_,k_,v_) .+ q
    
    # ffn
    q_ = relu.(el.ffn(el.n2(q))) .+ q
end
###########################################
############## DECODER LAYER ##############
###########################################
struct DecoderLayer
    n1::LayerNorm
    q1_mlp::Linear
    k1_mlp::Linear
    v1_mlp::Linear
    
    n2::LayerNorm
    q2_mlp::Linear
    k2_mlp::Linear
    v2_mlp::Linear
    
    n3::LayerNorm
    ffn::Linear
end
function DecoderLayer(token_size::Int)
    DecoderLayer(LayerNorm(token_size),    # n1
            Linear(token_size, token_size),# q1   
            Linear(token_size, token_size),# k1
            Linear(token_size, token_size),# v1
            LayerNorm(token_size),         # n2
            Linear(token_size, token_size),# q2
            Linear(token_size, token_size),# k2
            Linear(token_size, token_size),# v2
            LayerNorm(token_size),
            Linear(token_size, token_size))
end
#function (dl::DecoderLayer)(q::Knet.atype(), k::Knet.atype(), v::Knet.atype())
function (dl::DecoderLayer)(q, k, v)
    # self attn
    q_ = relu.(dl.q1_mlp(dl.n1(q)))
    k_ = relu.(dl.k1_mlp(dl.n1(q)))
    v_ = relu.(dl.v1_mlp(dl.n1(q)))
    q = self_attn(q_,k_,v_, mask=true) .+ q
    
    # cross attn
    q_ = relu.(dl.q2_mlp(dl.n2(q)))
    k_ = relu.(dl.k2_mlp(dl.n2(k)))
    v_ = relu.(dl.v2_mlp(dl.n2(v)))
    q = cross_attn(q_,k_,v_) .+ q
    
    # ffn
    q_ = relu.(dl.ffn(dl.n2(q))) .+ q
end


###########################################
############### TRANSFORMER ###############
###########################################
struct Transformer; imgEnc::ImageEncoder; sentEnc::SentEncoder; encL::EncoderLayer; decL::DecoderLayer; project::Linear; end
function Transformer(token_size::Int, vocab_size::Int)
    imgEnc = ImageEncoder(token_size)
    sentEnc = SentEncoder(token_size, vocab_size)

    encL = EncoderLayer(token_size)
    decL = DecoderLayer(token_size)

    project = Linear(token_size, vocab_size)
    
    Transformer(imgEnc, sentEnc, encL, decL, project)
end
#function (t::Transformer)(batch_imgs::KnetArray{Float32, 4}, batch_indices::Matrix{Int64})
function (t::Transformer)(batch_imgs, batch_indices)
    img_tokens = t.imgEnc(batch_imgs)
    print(typeof(img_tokens))
    img_tokens = t.encL(img_tokens, img_tokens, img_tokens)
    kv = img_tokens

    # Decoder
    word_tokens = t.sentEnc(batch_indices)
    q = word_tokens
    updated_q = t.decL(q, kv, kv)

    # Out
    word_probs = t.project(updated_q)
end
#function (t::Transformer)(batch_imgs::KnetArray{Float32, 4}, batch_indices::Matrix{Int64}, labels::Matrix{Int64})
function (t::Transformer)(batch_imgs, batch_indices, labels)
    word_probs = t(batch_imgs, batch_indices)
    nll(word_probs, batch_indices)
end
#function (t::Transformer)(batch_imgs::KnetArray{Float32, 4})
function (t::Transformer)(batch_imgs)
    # This function is used to predict a caption for a single image, as inference.
    batch_indices = reshape([tgt_iterator.vocab.eos],(1,1))
    
    while size(batch_indices,1) < 20
        word_probs = t(batch_imgs, batch_indices)
        word_probs = softmax(word_probs; dims=1)
        arg_max = argmax(word_probs[:,end,1]; dims=1)[1]
        batch_indices = cat(batch_indices, arg_max; dims=1)
        if batch_indices[end,1] == tgt_iterator.vocab.eos
            break
        end
    end
    img_tensor = permutedims(batch_imgs[:,:,:,1],(3,1,2))
    return img_tensor, batch_indices
end

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
    transformer = Transformer(128, length(tgt_iterator.vocab.i2w))

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