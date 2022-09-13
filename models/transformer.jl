using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random, CUDA, IterTools, DelimitedFiles, Statistics
Knet.atype() = KnetArray{Float32}
using Images, TestImages, OffsetArrays, Colors
using Plots

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

function (l::Embed)(x; p=0.1)
    # Your code here
    embeddings = l.w[:,x]
    embeddings = selu.(embeddings)
    dropout(embeddings, p)
end

###########################################
################# LINEAR ##################
###########################################
struct Linear; w; b; end

function Linear(inputsize::Int, outputsize::Int)
    Linear(param(outputsize,inputsize),param0(outputsize))
end

function (l::Linear)(x; p=0.1)
    #l.w*x .+ l.b
    if length(size(x)) < 3
        result = l.w * mat(x, dims = 1) .+ l.b
    else
        in_feats, num_t, b = size(x)
        out_feats = size(l.w, 1)
        result = l.w * reshape(x, (in_feats, num_t*b)) .+ l.b
        result = reshape(result, (out_feats, num_t, b))
    end
    result = selu.(result)
    result = dropout(result, p)
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
function (c::Preconv)(x; p=0.1)
    #temp = pool(conv4(c.w[1], x))# .+ w[2]))
    #a = pool(conv4(c.w[3], temp))# .+ w[4]))
    temp = pool(selu.(conv4(c.w[1], x)))# .+ w[2]))
    a = pool(selu.(conv4(c.w[3], temp)))# .+ w[4]))
    a = selu.(a)
    a = dropout(a, p)
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
function (r::MyResNet)(x; p=0.1)
    resnet_feats = ResNetLib.resnet50(r.w, r.m, x; stage=1) # 99x99x256*B
    resnet_feats=dropout(resnet_feats, p)
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
    
    dim_feats = Float32(sqrt(size(q,1))) #Float32(sqrt(128))
    a = softmax(a; dims=1)./dim_feats
    updated_q = bmm(v,a) # 128 x num_q x B
end

###########################################
############### CROSS ATTN ################
###########################################
#function cross_attn(q::Knet.atype(), k::Knet.atype(), v::Knet.atype())
function cross_attn(q, k, v)
    a = bmm(permutedims(k,(2,1,3)),q) # num_kv x num_q x B
    dim_feats = Float32(sqrt(size(q,1))) #Float32(sqrt(128))
    a = softmax(a; dims=1)./dim_feats
    updated_q = bmm(v,a) # 128 x num_q x B
end

###########################################
############### LAYER NORM ################
###########################################
struct LayerNorm; a; b; end
function LayerNorm(token_size::Int)
    LayerNorm(param(token_size,1), param0(token_size,1))
end
#function (ln::LayerNorm)(x::Knet.atype())
function (ln::LayerNorm)(x; p=0.1)
    avgs = mean(x, dims=1)
    stds = std(x, dims=1)
    normalized_x = ln.a .* (x .- avgs) ./ (stds .+ 1e-12) .+ ln.b
    dropout(normalized_x, p)
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
function (i::ImageEncoder)(x; p=0.1)
    # takes a batch of images 400x400x3xB
    # return tokens token_size x num_t x B
    resnet_feats = i.resnet(x; p=p) # 99x99x256*B
    precoder_output = i.preconv(resnet_feats; p=p) # 20x20x1xB
    h, w, c, b = size(precoder_output)
    temp = reshape(precoder_output, (h*w*c,b)) # 20*20*1xB
    o = i.precoder(temp; p=p) # 512xB (all tokens in a single vector x B)
    F, B = size(o) # F(whole feature vector size 512) B(batch size)
    img_tokens = reshape(o,(i.token_size, div(F,i.token_size), B)) # 128 x num_q x B
    img_tokens .+ oneD_PE(img_tokens)
    img_tokens = dropout(img_tokens, p)
end

###########################################
############ IMAGE ENCODER MLP ############
###########################################
struct ImageEncoderMLP; patch_size; token_size; linear; end
function ImageEncoderMLP(patch_size::Int, token_size::Int)
    ImageEncoderMLP(patch_size, token_size, Linear(patch_size*patch_size*3, token_size))
end
function (i::ImageEncoderMLP)(x; p=0.1)
    # takes a batch of images 400x400x3xB
    # return tokens token_size x num_t x B
    img_size = size(x, 1)
    batch_size = size(x, 4)
    
    num_tokens = div(img_size, i.patch_size) * div(img_size, i.patch_size) # the number of tokens from 1 image
    num_feats_in = i.patch_size * i.patch_size * 3 # each patch, with RGB channels, needs to turns into a flat input vector
    img_tokens = reshape(x, (num_feats_in, num_tokens, batch_size))
    img_tokens = i.linear(img_tokens; p=p)
    img_tokens .+ oneD_PE(img_tokens)
    img_tokens = dropout(img_tokens, p)
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

    pe[1:2:token_size,:,:] = sin.(pi[1:2:token_size,:,:].*w)
    pe[2:2:token_size,:,:] = cos.(pi[2:2:token_size,:,:].*w)
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
function (s::SentEncoder)(x; p=0.1)
    # takes a batch of worindices as input num_t x B
    # returns embeddings token_size x num_t x B
    word_tokens = s.embed(x; p=p)
    word_tokens .+ oneD_PE(word_tokens)
    word_tokens = dropout(word_tokens, p)
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
function (el::EncoderLayer)(q, k, v; p=0.1)
    # self attn
    q_ = selu.(el.q_mlp(el.n1(q; p=p); p=p))
    k_ = selu.(el.k_mlp(el.n1(k; p=p); p=p))
    v_ = selu.(el.v_mlp(el.n1(v; p=p); p=p))
    q = self_attn(q_,k_,v_) .+ q
    q = dropout(q, p)
    
    # ffn
    q_ = selu.(el.ffn(el.n2(q; p=p); p=p)) .+ q
    q_ = dropout(q_, p)
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
function (dl::DecoderLayer)(q, k, v; p=0.1)
    # self attn
    q_ = selu.(dl.q1_mlp(dl.n1(q; p=p); p=p))
    k_ = selu.(dl.k1_mlp(dl.n1(q; p=p); p=p))
    v_ = selu.(dl.v1_mlp(dl.n1(q; p=p); p=p))
    q = self_attn(q_,k_,v_, mask=true) .+ q
    q = dropout(q, p)
    
    # cross attn
    q_ = selu.(dl.q2_mlp(dl.n2(q; p=p); p=p))
    k_ = selu.(dl.k2_mlp(dl.n2(k; p=p); p=p))
    v_ = selu.(dl.v2_mlp(dl.n2(v; p=p); p=p))
    q = cross_attn(q_,k_,v_) .+ q
    q = dropout(q, p)
    
    # ffn
    q_ = selu.(dl.ffn(dl.n2(q; p=p); p=p)) .+ q
    q_ = dropout(q_, p)
end


###########################################
############### TRANSFORMER ###############
###########################################
#struct Transformer; imgEnc::ImageEncoderMLP; sentEnc::SentEncoder; encL::EncoderLayer; decL::DecoderLayer; project::Linear; eos::Int; end
struct Transformer; imgEnc; sentEnc::SentEncoder; encL::EncoderLayer; decL::DecoderLayer; project::Linear; eos::Int; end
function Transformer(token_size::Int, vocab_size::Int, eos::Int, use_conv=true)
    if use_conv
        imgEnc = ImageEncoder(token_size)
    else
        imgEnc = ImageEncoderMLP(20, token_size)
    end
    sentEnc = SentEncoder(token_size, vocab_size)

    encL = EncoderLayer(token_size)
    decL = DecoderLayer(token_size)

    project = Linear(token_size, vocab_size)
    
    Transformer(imgEnc, sentEnc, encL, decL, project, eos)
end
#resnet
#function (t::Transformer)(batch_imgs::KnetArray{Float32, 4}, batch_indices::Matrix{Int64})
function (t::Transformer)(batch_imgs, batch_indices; p=0.1)
    img_tokens = t.imgEnc(batch_imgs; p=p)
    img_tokens = t.encL(img_tokens, img_tokens, img_tokens; p=p)
    kv = img_tokens

    # Decoder
    word_tokens = t.sentEnc(batch_indices; p=p)
    q = word_tokens
    updated_q = t.decL(q, kv, kv; p=p)

    # Out
    word_probs = t.project(updated_q; p=p)
    word_probs = dropout(word_probs, p)
end
#function (t::Transformer)(batch_imgs::KnetArray{Float32, 4}, batch_indices::Matrix{Int64}, labels::Matrix{Int64})
function (t::Transformer)(batch_imgs, batch_indices, labels; p=0.1, use_smooth_loss=false)
    word_probs = t(batch_imgs, batch_indices; p=p)
    if use_smooth_loss
        return nll(word_probs, batch_indices)
    else
        return nll(word_probs, batch_indices)
    end
        
end
#function (t::Transformer)(batch_imgs::KnetArray{Float32, 4})
function (t::Transformer)(batch_imgs; p=0.1)
    # This function is used to predict a caption for a single image, as inference.
    batch_indices = reshape([t.eos],(1,1))
    
    while size(batch_indices,1) < 20
        word_probs = t(batch_imgs, batch_indices; p=p)
        #word_probs = softmax(word_probs; dims=1)
        arg_max = argmax(word_probs[:,end,1]; dims=1)[1]
        batch_indices = cat(batch_indices, arg_max; dims=1)
#         if batch_indices[end,1] == t.eos
#             break
#         end
    end
    img_tensor = permutedims(batch_imgs[:,:,:,1],(3,1,2))
    return img_tensor, batch_indices
end

###########################################
############### LR WARMUP SCHEDULE ########
###########################################
function compute_lr(d_model, step_num; warmup_steps=4000)
    1/Float32(sqrt(d_model)) * min(1/Float32(sqrt(step_num)), step_num*1/Float32(sqrt(warmup_steps)^3))
end


###########################################
############### LABEL SMOOTHING ###########
###########################################
function findindices(scores, labels::AbstractArray{<:Integer}; dims=1)
    ninstances = length(labels)
    nindices = 0
    indices = Vector{Int}(undef,ninstances)
    if dims == 1                   # instances in first dimension
        y1 = size(scores,1)
        y2 = div(length(scores),y1)
        if ninstances != y2; throw(DimensionMismatch()); end
        @inbounds for j=1:ninstances
            if labels[j] == 0; continue; end
            indices[nindices+=1] = (j-1)*y1 + labels[j]
        end
    elseif dims == 2               # instances in last dimension
        y2 = size(scores,ndims(scores))
        y1 = div(length(scores),y2)
        if ninstances != y1; throw(DimensionMismatch()); end
        @inbounds for j=1:ninstances
            if labels[j] == 0; continue; end
            indices[nindices+=1] = (labels[j]-1)*y1 + j
        end
    else
        error("findindices only supports dims = 1 or 2")
    end
    return (nindices == ninstances ? indices : view(indices,1:nindices))
end

function label_smoothed_cross_entropy(scores, labels; epsilon=0.1)
    
    num_classes, num_word, num_sent = size(scores)
    smooth_labels = zeros(size(scores))

    negative_value = epsilon / (num_classes - 1)
    positive_value = 1.0 - epsilon

    smooth_labels .= negative_value

    smooth_labels[findindices(smooth_labels, labels, dims=1)] .= positive_value

    mask = Array(labels.==0)
    smooth_labels = permutedims(smooth_labels, (2,3,1))
    smooth_labels[mask,:] .= 0
    smooth_labels = permutedims(smooth_labels, (3,1,2))
    
    # now smooth_labels and scores have same shape: (vocab size, sentence length, batch size(number of sentences in batch))
    # now lets compute an actual loss value
    word_probs = logsoftmax(scores; dims=1)
    
    mean(sum(-1.0 .* Knet.atype(smooth_labels) .* word_probs, dims=1)) 
    
end