require "torch"
require "math"
local word_encoder_decoder = require  "word_encoder_decoder"
local label_encoder_decoder = require  "label_encoder_decoder"
local voc_size = 300000
local batchs_per_file = 100000
local batch_size = 8
local nlppl = 5 -- negative labels per positive label

local data_file = "./f60.txt"

local wed = word_encoder_decoder.create("vocabulary.txt",voc_size)
local led = label_encoder_decoder.create("labels.txt")


function split(str, delim)
    -- Eliminate bad cases...
    if string.find(str, delim) == nil then
        return { str }
    end
    local result = {}
    local pat = "(.-)" .. delim .. "()"
    local lastPos
    for part, pos in string.gfind(str, pat) do
        table.insert(result, part)
        lastPos = pos
    end
    table.insert(result, string.sub(str, lastPos))
    return result
end

function shuffle(t)
  local n = #t
  while n >= 2 do
    -- n is now the last pertinent index
    local k = math.random(n) -- 1 <= k <= n
    -- Quick swap
    t[n], t[k] = t[k], t[n]
    n = n - 1
  end
  return t
end

local label_dist = {}
local f = io.open("label_dist_bin.txt", "r")
for line in f:lines() do
    local temp = split(line," ")
    local label = temp[1]
    local dist =  temp[2] + 0
    table.insert(label_dist,{[1]=label,[2]=dist})
end
f:close()

function sample_label()
  local num = math.random(1, label_dist[#label_dist][2]-1)
  for k = 1, #label_dist do
    if num < label_dist[k][2] then
      return label_dist[k][1]
    end
  end
end

function is_in(label,labels)
  for l = 1, #labels do
    if labels[l] == label then
      return true
    end
  end
  return false
end

function sample_negatives(labels,nlppl)
  local result = {}
  for i = 1, nlppl do
    local cand = sample_label()
    while is_in(cand,labels) do
      cand = sample_label()
    end
    table.insert(result,cand)
  end
  return result
end

local s = 1
function save_t(corpus,batchs_per_file,batch_size)
  local data = {}
  for i = 1,batchs_per_file do
    local batch = {}
    for j = 1, batch_size do
      table.insert(batch,corpus[(i-1)*batch_size+j])
    end
    table.insert(data,batch)
  end
  data = torch.Tensor(data)
  torch.save("data/length60."..s..".data.t7",data)
  s = s + 1
  print("tin.tin.tin....")
end

local j = 0
local f = io.open(data_file, "r")
local corpus = {}

for line in f:lines() do
    local temp = split(line," \t")
    local sentence = split(temp[2]," ")
    local labels = split(temp[1]," ")
    for l = 1, #labels do
        local positive_label = labels[l]
        local negative_labels = sample_negatives(labels,nlppl)
        for nl = 1, nlppl do
          local negative_label = negative_labels[nl]
          local x = {}
          for wn = 1, #sentence - 1 do
            local word = sentence[wn]
            table.insert(x, wed:Encode(word))
          end
          table.insert(x, led:Encode(positive_label))
          table.insert(x, led:Encode(negative_label))
          table.insert(corpus,x)
          j = j + 1
          if j % (batchs_per_file * batch_size) == 0 then
            courpus = shuffle(corpus)
            save_t(corpus,batchs_per_file,batch_size)
            corpus = {}
            collectgarbage("collect")
          end
        end
    end
end
if corpus ~= {} then
  save_t(corpus,#corpus/batch_size,batch_size)
end
f:close()
