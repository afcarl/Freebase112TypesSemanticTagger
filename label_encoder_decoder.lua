local label_encoder_decoder = {}


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

function label_encoder_decoder.create(file_name)
  local self = {}
  setmetatable(self,Encoder_decoder)
  self.encoder = {}
  self.decoder = {}
  print("loading label files...")
  local f = io.open(file_name, "r")
  for line in f:lines() do
      local temp = split(line," ")
      local id = 0 + temp[1] -- integer cast
      local label = temp[2]
      self.encoder[label] = id
      self.decoder[id] = label

  end
  f:close()
  self.size = size

  function self:Encode(label)
    local id = self.encoder[label]
    return id
  end
      ---- Decoder's :decode method
  function self:Decode(id)
    local label = self.decoder[id]
    return label
  end

  return self
end

return label_encoder_decoder
