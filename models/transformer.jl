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
struct Transformer; imgEnc::ImageEncoder; sentEnc::SentEncoder; encL::EncoderLayer; decL::DecoderLayer; project::Linear; eos::Int; end
function Transformer(token_size::Int, vocab_size::Int, eos:: Int)
    imgEnc = ImageEncoder(token_size)
    sentEnc = SentEncoder(token_size, vocab_size)

    encL = EncoderLayer(token_size)
    decL = DecoderLayer(token_size)

    project = Linear(token_size, vocab_size)
    
    Transformer(imgEnc, sentEnc, encL, decL, project, eos)
end
#resnet
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
    batch_indices = reshape([t.eos],(1,1))
    
    while size(batch_indices,1) < 20
        word_probs = t(batch_imgs, batch_indices)
        word_probs = softmax(word_probs; dims=1)
        arg_max = argmax(word_probs[:,end,1]; dims=1)[1]
        batch_indices = cat(batch_indices, arg_max; dims=1)
        if batch_indices[end,1] == t.eos
            break
        end
    end
    img_tensor = permutedims(batch_imgs[:,:,:,1],(3,1,2))
    return img_tensor, batch_indices
end