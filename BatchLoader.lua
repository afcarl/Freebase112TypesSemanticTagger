require 'torch'

local BatchLoader = {}
BatchLoader.__index = BatchLoader

function dirLookup(dir)
   local p = io.popen('find "'..dir..'" -type f')  --Open directory look for files, save data in p. By giving '-type f' as parameter, it returns all files.
   local result = {}
   for file in p:lines() do                         --Loop through all files
       table.insert(result,file)
   end
   return result
end

function BatchLoader.create(dir)
    local self = {}
    setmetatable(self, BatchLoader)
    self.file_names = dirLookup(dir)
    self.file_num = 2
    self.dataset = torch.load(self.file_names[self.file_num])
    self.batch_num = 0
    self.size = self.dataset:size(1)
    self.batch_size = self.dataset:size(2)
    self.seq_length = self.dataset:size(3)
    return self
  end


function BatchLoader:next_batch()
  self.batch_num = self.batch_num + 1
  if self.batch_num > self.size then
    self.file_num = (self.file_num % #self.file_names) + 1
    self.dataset = torch.load(self.file_names[self.file_num])
    self.batch_num = 1
    self.size = self.dataset:size(1)
    self.batch_size = self.dataset:size(2)
    self.seq_length = self.dataset:size(3)
  end
  local x = self.dataset[{self.batch_num,{},{1,self.seq_length-2}}]
  local yp = self.dataset[{self.batch_num,{},self.seq_length-1}]
  local yn = self.dataset[{self.batch_num,{},self.seq_length}]
  return x,yp,yn
end

return BatchLoader
