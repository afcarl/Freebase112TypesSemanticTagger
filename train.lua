require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
                    -- class name is Embedding (not namespaced)
local model_utils=require 'model_utils'
local BatchLoader = require 'BatchLoader'
local LSTM = require "LSTM"


local loader = BatchLoader.create("./data/")
local rnn_size = 40
local class_embed_size = 80
local seq_length = 40
local batch_size = 8
local in_size = 300000
local class_size = 113
local iterations = 310000

local model = torch.load("./model.t7")

--local test_loader = BatchLoader.create("../data/test_dataset.t7","../data/dataset_info.t7")




function Attention(x_size)
    local x = nn.Identity()()
    local hx = nn.Linear(x_size,x_size)(x)
    local t = nn.Tanh()(hx)
    local h = nn.Linear(x_size,1)({t})
    local o = nn.Exp()(h)
    return nn.gModule({x},{o})
end

function Bilinear(x_size,y_size)
    local x = nn.Identity()()
    local y = nn.Identity()()
    local h = nn.Linear(x_size,y_size)(x)
    local hy = nn.CMulTable()({h,y})
    local o = nn.Sum(2)(hy)
    return nn.gModule({x,y},{o})
end

function Dotproduct(x_size)
    local x = nn.Identity()()
    local y = nn.Identity()()
    local hy = nn.CMulTable()({x,y})
    local o = nn.Sum(2)(hy)
    return nn.gModule({x,y},{o})
end

function Score(x_size,y_size)
    local x = nn.Identity()()
    local y = nn.Identity()()
    local xy = nn.JoinTable(2)({x,y})
    local h1 = nn.Linear(x_size+y_size,x_size+y_size)(xy)
    local z1  = nn.ReLU()(h1)
    local h2 = nn.Linear(x_size+y_size,x_size+y_size)(z1)
    local z2 = nn.ReLU()(h2)
    local o = nn.Linear(x_size+y_size,1)(z2)
    return nn.gModule({x,y},{o})
end

if model then
  protos = model.protos
  label_embed  = model.label_embed
  out_layer = model.out_layer
else
  protos = {}
  protos.embed = Embedding(in_size, rnn_size)
  protos.lstmF = LSTM.lstm(rnn_size)
  protos.lstmB = LSTM.lstm(rnn_size)
  protos.concat = nn.JoinTable(2)
  protos.attention = Attention(rnn_size*2)

  label_embed = Embedding(class_size,class_embed_size)
  out_layer = Dotproduct(2*rnn_size)
  --local out_layer = Score(2*rnn_size,class_embed_size)
end



local normalize =  nn.Normalize(1)
local mixture = nn.MixtureTable()


local params, grad_params = model_utils.combine_all_parameters(protos.embed, protos.lstmF, protos.lstmB,protos.attention,label_embed,out_layer)

---- CLONE -----
local clones = {}
for name,proto in pairs(protos) do
    clones[name] = model_utils.clone_many_times(proto, seq_length)
end

local positive_embed,negative_embed = unpack(model_utils.clone_many_times(label_embed,2))
local positive_out_layer, negative_out_layer = unpack(model_utils.clone_many_times(out_layer,2))
local criterion = nn.MarginRankingCriterion(1)

if not model then
  params:uniform(-1e-2, 1e-2)
end

local initstate_c = torch.zeros(batch_size, rnn_size)
local initstate_h = initstate_c:clone()
local d_finalstate_c = torch.zeros(batch_size, rnn_size)
local d_finalstate_h = initstate_c:clone()


function feval(params_)
    local data,positive_target,negative_target = loader:next_batch()

    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()

    local embeddings = {}
    local lstm_cF = {[0]=initstate_c}
    local lstm_cB = {[seq_length+1]=initstate_c}
    local lstm_hF = {[0]=initstate_h}
    local lstm_hB = {[seq_length+1]=initstate_h}
    local concats = {}
    local atts = torch.zeros(batch_size,seq_length)

    -- WORD EMBEDDING
    for t=1,seq_length do
        embeddings[t] = clones.embed[t]:forward(data[{{}, t}])
    end
    -- LSTM FORWARD
    for t=1,seq_length do
        lstm_cF[t], lstm_hF[t] = unpack(clones.lstmF[t]:forward{embeddings[t], lstm_cF[t-1], lstm_hF[t-1]})
    end
    -- LSTM BACKWARD
    for t=seq_length,1,-1 do
        lstm_cB[t], lstm_hB[t] = unpack(clones.lstmB[t]:forward{embeddings[t], lstm_cB[t+1], lstm_hB[t+1]})
    end
    -- CONCATENATION
    for t=1,seq_length do
        concats[t] = clones.concat[t]:forward({lstm_hF[t],lstm_hB[t]})
    end
    -- ATTENTION
    for t=1,seq_length do
        atts[{{},t}] = clones.attention[t]:forward(concats[t])
    end
    -- NORMALIZE
    local normalized_atts = normalize:forward(atts)
    -- MIXTURE
    local mix = mixture:forward({normalized_atts,concats})
    -- POSITIVE  LABEL EMBEDDING
    local positive_label_embedding = positive_embed:forward(positive_target)
    -- POSITIVE SCORE
    local positive_score = positive_out_layer:forward({mix,positive_label_embedding})
    -- NEGATIVE LABEL EMBEDDING
    local negative_label_embedding = negative_embed:forward(negative_target)
    -- NEGATIVE SCORE
    local negative_score = negative_out_layer:forward({mix,negative_label_embedding})
    -- LOSS
    local loss = criterion:forward({positive_score,negative_score},1)

    -------------------------- BACKWARD --------------------
    --d POSITIVE_SCORE,   d NEGATIVE_SCORE
    local d_positive_score, d_negative_score = unpack(criterion:backward({positive_score,negative_score},1))
    -- d MIXTURE, d POSITIVE LABEL EMBEDDING
    local temp = positive_out_layer:backward({mix, positive_label_embedding},d_positive_score)
    local d_mix = temp[1]
    local d_positive_label_embedding = temp[2]
    -- d MIXTURE, d NEGATIVE LABEL EMBEDDING
    local temp = negative_out_layer:backward({mix, negative_label_embedding},d_negative_score)
    local d_mix = d_mix + temp[1]
    local d_negative_label_embedding = temp[2]
    -- d NORMALIZED ATTENTION, d CONCATENATION
    local temp = mixture:backward({normalized_atts,concats},d_mix)
    local d_normalized_atts = temp[1]
    local d_concats = temp[2]
    -- d ATTENTION
    local d_atts = normalize:backward(atts,d_normalized_atts)
    -- d CONCATS
    for t = 1,seq_length do
        d_concats[t] = d_concats[t] + clones.attention[t]:backward(concats[t],d_atts[{{},t}])
    end
    -- d LSTM FORWARD, d LSTM BACKWARD
    local d_embeddings = {}
    local d_lstm_cF = {[seq_length]=d_finalstate_c}
    local d_lstm_cB = {[1]=d_finalstate_c}
    local d_lstm_hF = {}
    local d_lstm_hB = {}

    for t = 1,seq_length do
          d_lstm_hF[t], d_lstm_hB[t] = unpack(clones.concat[t]:backward({lstm_hF[t],lstm_hB[t]},d_concats[t]))
    end
    for t = seq_length,1,-1 do
         local temp = clones.lstmF[t]:backward(
            {embeddings[t], lstm_cF[t-1], lstm_hF[t-1]},
            {d_lstm_cF[t], d_lstm_hF[t]}
            )
         d_embeddings[t] = temp[1]
         d_lstm_cF[t-1] = temp[2]
         d_lstm_hF[t-1] = d_lstm_hF[t] + temp[3]
    end
    for t = 1,seq_length do
         local temp = clones.lstmB[t]:backward(
            {embeddings[t], lstm_cB[t+1], lstm_hB[t+1]},
            {d_lstm_cB[t], d_lstm_hB[t]}
            )
         d_embeddings[t] = d_embeddings[t] + temp[1]
         d_lstm_cB[t+1] = temp[2]
         d_lstm_hB[t+1] = d_lstm_hB[t] + temp[3]
    end
    for t = 1,seq_length do
         clones.embed[t]:backward(data[{{}, t}], d_embeddings[t])
    end

    positive_embed:backward(positive_target,d_positive_label_embedding)
    negative_embed:backward(negative_target,d_negative_label_embedding)
    grad_params:clamp(-5, 5)
    return loss, grad_params
end


-- optimization stuff
local LOSS = 0
local optim_state = {learningRate = 0.0001,momentum = 0.9}
print(iterations);
for i = 1, iterations do
    local _, loss = optim.adam(feval, params, optim_state)
    print("(iteration: "..i..") "..torch.sum(loss[1])/batch_size);
    LOSS = LOSS + torch.sum(loss[1])
    if i % 10000 == 0 then
      print(LOSS/(batch_size*10000))
      torch.save("./model.t7",model)
      print("saved..")
    end
end

local model = {protos=protos, label_embed=label_embed,out_layer=out_layer}
torch.save("./model.t7",model)
