local word_encoder_decoder = {}


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

function word_encoder_decoder.create(file_name,size)
  local self = {}
  setmetatable(self,Encoder_decoder)
  self.encoder = {}
  self.encoder["__UNKNOWN__"] = size - 3
  self.encoder["__PAD__"] = size - 2
  self.encoder["__START__"] =  size - 1
  self.encoder["__END__"] = size
  self.decoder = {}
  self.decoder[size-3] = "__UNKNOWN__"
  self.decoder[size-2] = "__PAD__"
  self.decoder[size-1] = "__START__"
  self.decoder[size] = "__END__"
  print("loading vocabulary files...")
  local f = io.open(file_name, "r")
  for line in f:lines() do
      local temp = split(line," ")
      local id = 0 + temp[1] -- integer cast
      local word = temp[2]
      if id < size - 1 then
        self.encoder[word] = id
        self.decoder[id] = word
      end
  end
  f:close()
  self.size = size

  function self:Encode(word)
      local id = self.encoder[word]
      --- word that is not in vocabulary is treated as __UNKNOWN__.
      if id then return id else return size - 3 end
  end
      ---- Decoder's :decode method
  function self:Decode(id)
      local word = self.decoder[id]
      --- word that is not in vocabulary is treated as __UNKNOWN__.
      if word then return word else return "__UNKNOWN__" end
  end

  return self
end


return word_encoder_decoder
