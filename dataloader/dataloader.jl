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
        img_name = split(readline(s))[1] #2
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

