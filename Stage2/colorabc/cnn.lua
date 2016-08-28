----------------------------------------------------------------------
-- This script shows how to train different models on the MNIST
-- dataset, using multiple optimization techniques (SGD, LBFGS)
--
-- This script demonstrates a classical example of training
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem.
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nnx'
require 'cutorch'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
cjson = require 'cjson'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
-s,--save          (default "logs")      subdirectory to save logs
-n,--network       (default "")          reload pretrained network
-m,--model         (default "convnet")   type of model tor train: convnet | mlp | linear
-f,--full                                use the full dataset
--path             (string)              output path
-o,--optimization  (default "SGD")       optimization: SGD | LBFGS
-r,--learningRate  (default 0.05)        learning rate, for SGD only
-b,--batchSize     (default 10)          batch size
-m,--momentum      (default 0)           momentum, for SGD only
-i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
-d,--dropout       (default 0.5)         Dropout
--coefL1           (default 0)           L1 penalty on the weights
--coefL2           (default 0)           L2 penalty on the weights
-t,--threads       (default 4)           number of threads
]]

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats, for SGD
if opt.optimization == 'SGD' then
  torch.setdefaulttensortype('torch.FloatTensor')
end

-- batch size?
if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
  error('LBFGS should not be used with small mini-batches; 1000 is recommended')
end

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {}
for i = 1,26 do
  classes[i] = string.format(i)
end

-- geometry: width and height of input images
geometry = {128, 128}

if opt.network == '' then
  -- define model to train
  model = nn.Sequential()

  -- current best: image(3, 128, 128) -> conv(3, 64, 5, 5) -> pool(3, 3, 3, 3,
  -- 1, 1) -> conv(64, 64, 3, 3) -> pool(3, 3, 3, 3, 1, 1) -> dropout(0.25) ->
  -- linear(64*14*14, 512) -> dropout(0.5) -> linear(512, 26)
  if opt.model == 'convnet' then
    ------------------------------------------------------------
    -- convolutional network
    ------------------------------------------------------------
    -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
    --model:add(nn.SpatialConvolution(3, 32, 5, 5))
    model:add(nn.SpatialConvolution(3, 64, 5, 5))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(3, 3, 3, 3, 1, 1))
    --model:add(nn.Dropout(opt.dropout))
    -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
    model:add(nn.SpatialConvolution(64, 64, 3, 3))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(3, 3, 3, 3, 1, 1))
    --model:add(nn.SpatialConvolution(32, 32, 3, 3))
    --model:add(nn.ReLU())
    --model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    model:add(nn.Dropout(0.25))
    -- stage 4 : standard 2-layer MLP:
    model:add(nn.Reshape(64*14*14))
    model:add(nn.Linear(64*14*14, 512))
    model:add(nn.ReLU())
    model:add(nn.Dropout(opt.dropout))
    model:add(nn.Linear(512, #classes))
    ------------------------------------------------------------

  elseif opt.model == 'mlp' then
    ------------------------------------------------------------
    -- regular 2-layer MLP
    ------------------------------------------------------------
    model:add(nn.Reshape(1024))
    model:add(nn.Linear(1024, 2048))
    model:add(nn.ReLU())
    model:add(nn.Linear(2048, #classes))
    ------------------------------------------------------------

  elseif opt.model == 'linear' then
    ------------------------------------------------------------
    -- simple linear model: logistic regression
    ------------------------------------------------------------
    model:add(nn.Reshape(1024))
    model:add(nn.Linear(1024, #classes))
    ------------------------------------------------------------

  else
    print('Unknown model type')
    cmd:text()
    error()
  end
else
  print('<trainer> reloading previously trained network')
  model = torch.load(opt.network)
end


-- verbose
print('<alphabet> using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
model = model:cuda()
-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

criterion = nn.ClassNLLCriterion():cuda()

----------------------------------------------------------------------
-- get/create dataset
--
function loadData()
  local letter_cnt = cjson.decode(io.open('../letter_cnt.json', 'r'):read())
  local train_dataset, dev_dataset = {}, {}
  train_dataset.data = {}
  train_dataset.labels = {}
  dev_dataset.data = {}
  dev_dataset.labels = {}
  for ascii = 97, 122 do
    collectgarbage()
    local c = string.char(ascii)
    for i = 2, letter_cnt[c] do
      local img = image.load(string.format('../alphabet_data/test/%s%d.jpg', c, i))
      img = image.rgb2hsv(img)
      --local ud_img = image.vflip(image.hflip(img))
      for x = 4, 5 do
        for y = 4, 5 do
          if torch.rand(1)[1] < 0.8 then
            for i = 0, 0 do
              table.insert(train_dataset.data,
                           image.scale(image.rotate(img[{{},{6*x+1,6*x+162},{6*y+1,6*y+162}}], i*90), geometry[1], geometry[2]):reshape(1, 3, geometry[1], geometry[2]))
              table.insert(train_dataset.labels, ascii - 96)
            end
            --table.insert(train_dataset.data,
                         --image.scale(img[{{},{6*x+1,6*x+162},{6*y+1,6*y+162}}], geometry[1], geometry[2]):reshape(1, 3, geometry[1], geometry[2]))
            --table.insert(train_dataset.data,
                         --image.scale(ud_img[{{},{6*x+1,6*x+162},{6*y+1,6*y+162}}], geometry[1], geometry[2]):reshape(1, 3, geometry[1], geometry[2]))
            --table.insert(train_dataset.labels, ascii - 96)
            --table.insert(train_dataset.labels, ascii - 96)
          else
            for i = 0, 0 do
              table.insert(dev_dataset.data,
                           image.scale(image.rotate(img[{{},{6*x+1,6*x+162},{6*y+1,6*y+162}}], i*90), geometry[1], geometry[2]):reshape(1, 3, geometry[1], geometry[2]))
              table.insert(dev_dataset.labels, ascii - 96)
            end
            --table.insert(dev_dataset.data,
                         --image.scale(img[{{},{6*x+1,6*x+162},{6*y+1,6*y+162}}], geometry[1], geometry[2]):reshape(1, 3, geometry[1], geometry[2]))
            --table.insert(dev_dataset.data,
                         --image.scale(ud_img[{{},{6*x+1,6*x+162},{6*y+1,6*y+162}}], geometry[1], geometry[2]):reshape(1, 3, geometry[1], geometry[2]))
            --table.insert(dev_dataset.labels, ascii - 96)
            --table.insert(dev_dataset.labels, ascii - 96)
          end
        end
      end
    end
  end
  train_dataset.data = nn.JoinTable(1):forward(train_dataset.data)
  train_dataset.data = torch.cdiv(train_dataset.data - torch.mean(train_dataset.data, 1):expandAs(train_dataset.data), torch.std(train_dataset.data, 1):expandAs(train_dataset.data))
  dev_dataset.data = nn.JoinTable(1):forward(dev_dataset.data)
  dev_dataset.data = torch.cdiv(dev_dataset.data - torch.mean(dev_dataset.data, 1):expandAs(dev_dataset.data), torch.std(dev_dataset.data, 1):expandAs(dev_dataset.data))
  train_dataset.labels = torch.IntTensor(train_dataset.labels):cuda()
  dev_dataset.labels = torch.IntTensor(dev_dataset.labels):cuda()
  train_dataset.size = train_dataset.labels:size(1)
  dev_dataset.size = dev_dataset.labels:size(1)
  torch.save('maomao_hsv_mean.th', torch.mean(train_dataset.data, 1))
  torch.save('maomao_hsv_std.th', torch.std(train_dataset.data, 1))
  return train_dataset, dev_dataset
end
--if opt.full then
--nbTrainingPatches = 60000
--nbTestingPatches = 10000
--else
--nbTrainingPatches = 2000
--nbTestingPatches = 1000
--print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
--end

-- create training set and normalize
--trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
--trainData:normalizeGlobal(mean, std)

-- create test set and normalize
--testData = mnist.loadTestSet(nbTestingPatches, geometry)
--testData:normalizeGlobal(mean, std)

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- training function
function train(dataset, lr)
  model:training()
  -- epoch tracker
  epoch = epoch or 1

  local shuffle = torch.randperm(dataset.size):long()
  dataset.data = dataset.data:index(1, shuffle)
  dataset.labels = dataset.labels:index(1, shuffle)

  -- local vars
  local time = sys.clock()

  --local avg_f = 0

  -- do one epoch
  print('<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  for t = 1, dataset.size, opt.batchSize do
    -- create mini batch
    local batch_end = math.min(t+opt.batchSize-1, dataset.size)
    local inputs = dataset.data[{{t,batch_end},{},{},{}}]:cuda()
    local targets = dataset.labels[{{t,batch_end}}]

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
      -- just in case:
      collectgarbage()

      -- get new parameters
      if x ~= parameters then
        parameters:copy(x)
      end

      -- reset gradients
      gradParameters:zero()

      -- evaluate function for complete mini batch
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      --avg_f = avg_f + f

      -- estimate df/dW
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        -- locals:
        local norm,sign= torch.norm,torch.sign

        -- Loss:
        f = f + opt.coefL1 * norm(parameters,1)
        f = f + opt.coefL2 * norm(parameters,2)^2/2

        -- Gradients:
        gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
      end

      -- update confusion
      for i = 1, batch_end-t+1 do
        confusion:add(outputs[i], targets[i])
      end

      -- return f and df/dX
      return f,gradParameters
    end

    -- optimize on current mini-batch
    if opt.optimization == 'LBFGS' then

      -- Perform LBFGS step:
      lbfgsState = lbfgsState or {
        maxIter = opt.maxIter,
        lineSearch = optim.lswolfe
      }
      optim.lbfgs(feval, parameters, lbfgsState)

      -- disp report:
      print('LBFGS step')
      print(' - progress in batch: ' .. t .. '/' .. dataset.size)
      print(' - nb of iterations: ' .. lbfgsState.nIter)
      print(' - nb of function evalutions: ' .. lbfgsState.funcEval)

    elseif opt.optimization == 'SGD' then

      -- Perform SGD step:
      sgdState = sgdState or {
        learningRate = lr,
        momentum = opt.momentum,
        learningRateDecay = 5e-7
      }
      optim.sgd(feval, parameters, sgdState)

      -- disp progress
      xlua.progress(batch_end, dataset.size)

    elseif opt.optimization == 'adam' then

      -- Perform adam step:
      adamState = adamState or {
        learningRate = lr,
        learningRateDecay = 5e-7
      }
      optim.adam(feval, parameters, adamState)

      -- disp progress
      xlua.progress(batch_end, dataset.size)

    elseif opt.optimization == 'adadelta' then

      -- Perform adam step:
      adadeltaState = adadeltaState or {
        learningRate = lr,
        learningRateDecay = 5e-7
      }
      optim.adadelta(feval, parameters, adadeltaState)

      -- disp progress
      xlua.progress(batch_end, dataset.size)

    else
      error('unknown optimization method')
    end
  end
  xlua.progress(dataset.size, dataset.size)

  --avg_f = avg_f / dataset.size
  --print("\n<training> EEEEEEEEEEEEE avg_f = " .. avg_f)

  -- time taken
  time = sys.clock() - time
  time = time / dataset.size
  --print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  --print(confusion)
  --trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  --confusion:zero()

  ---- save/log current net
  --local filename = paths.concat(opt.save, 'alphabet.net')
  --os.execute('mkdir -p ' .. sys.dirname(filename))
  --if paths.filep(filename) then
    --os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
  --end
  --print('<trainer> saving network to '..filename)
  -- torch.save(filename, model)

  -- next epoch
  epoch = epoch + 1
end

-- test function
function test(dataset)
  model:evaluate()
  -- local vars
  local time = sys.clock()

  local acc = 0
  local avg_f = 0

  -- test over given dataset
  print('<trainer> on testing Set:')
  for t = 1,dataset.size,opt.batchSize do

    -- create mini batch
    local batch_end = math.min(t+opt.batchSize-1, dataset.size)
    local inputs = dataset.data[{{t,batch_end},{},{},{}}]:cuda()
    local targets = dataset.labels[{{t,batch_end}}]

    -- test samples
    local preds = model:forward(inputs)
    local f = criterion:forward(preds, targets)
    avg_f = avg_f + f

    for i = 1, batch_end-t+1 do
      local _, idx = preds[i]:max(1)
      if idx[1] == targets[i] then
        acc = acc + 1
      end
    end

    -- confusion:
    for i = 1, batch_end-t+1 do
      confusion:add(preds[i], targets[i])
    end
    -- disp progress
    xlua.progress(batch_end, dataset.size)
  end
  xlua.progress(dataset.size, dataset.size)

  acc = acc / dataset.size * 100
  avg_f = avg_f / dataset.size
  print("\n<dev> acc = " .. acc .. '%')
  print("<dev> loss = " .. avg_f)

  -- timing
  time = sys.clock() - time
  time = time / dataset.size
  --print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  --print(confusion)
  testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  confusion:zero()
  return acc, avg_f
end

----------------------------------------------------------------------
-- and train!
--
utils.printf('Number of parameters = %d\n', parameters:size(1))
train_dataset, dev_dataset = loadData()
local losses = {}
local anneal_idx = 1
local loss
local lr = opt.learningRate
local avg_loss
local best_model = nil
local best_acc = -1
local epoch = 1

while true do
  -- train/test
  train(train_dataset, lr)
  test(train_dataset)
  local acc, loss = test(dev_dataset)

  if acc > best_acc then
    best_model = model
    best_acc = acc
  end

  if #losses > 3 then
    avg_loss = (losses[#losses]+losses[#losses-1]+losses[#losses-2])/3
    if loss > avg_loss then
      lr = lr * 0.3
      print('Learning rate annealing ' .. anneal_idx)
      if anneal_idx >= 7 then
        break
      end
      anneal_idx = anneal_idx + 1
      losses = {}
    end
  end
  table.insert(losses, loss)

  if epoch >= 20 then
    break
  end
  epoch = epoch + 1

  -- plot errors
  --if opt.plot then
    --trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    --testLogger:style{['% mean class accuracy (test set)'] = '-'}
    --trainLogger:plot()
    --testLogger:plot()
  --end
end
torch.save(opt.path, best_model)
