--
--  Copyright (c) 2016, Manuel Araoz
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  classifies an image using a trained model
--

require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'

local t = require './transforms'
local imagenetLabel = require './cifar10'
npy4th = require 'npy4th'

if #arg < 2 then
  io.stderr:write('Usage: th classify.lua [MODEL] [NPY_PATH]...\n')
  os.exit(1)
end

-- Load the model
model = torch.load(arg[1]):cuda()
local softMaxLayer = cudnn.SoftMax():cuda()

-- add Softmax layer
model:add(softMaxLayer)

-- Evaluate mode
model:evaluate()

-- The model was trained with this input normalization
meanstd = {
  mean = {125.3/255, 123.0/255, 113.9/255},
  std  = {63.0/255,  62.1/255,  66.7/255},
}

transform = t.Compose{
  t.Scale(32),
  t.ColorNormalize(meanstd),
  t.CenterCrop(32),
}

local N = 1

function predict(img_array, n)
  -- Scale, normalize, and crop the image
  for i=1,n do
    img_array[i] = transform(img_array[i])
  end

  -- View as mini-batch of size 1
  --local batch = img:view(1, table.unpack(img:size():totable()))

  -- Get the output of the softmax
  local output = model:forward(img_array:cuda()):squeeze()

  -- Get the top 5 class indexes and probabilities
  local probs, indexes = output:topk(N, true, true)

  return indexes:squeeze()
  --print(indexes:size())

  --return indexes[1] 
end

while true do
  io.read()
  local big_array = npy4th.loadnpy(arg[2]):permute(1, 4, 2, 3)
  local n = big_array:size()[1]

  --c0 = sys.clock()
  local res = predict(big_array, n)
  --print(sys.clock()-c0)

  for i=1,n do
    --print('Classes for', i, ':', imagenetLabel[res[i]])
    io.write(res[i]-1, ' ')
  end
  io.write('\n')
  io.flush()
end
