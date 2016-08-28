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

geometry = {128, 128}
batchSize = 128

criterion = nn.ClassNLLCriterion():cuda()

--function loadData()
  --local letter_cnt = cjson.decode(io.open('../letter_cnt.json', 'r'):read())
  --local train_dataset, dev_dataset = {}, {}
  --train_dataset.data = {}
  --train_dataset.labels = {}
  --dev_dataset.data = {}
  --dev_dataset.labels = {}
  --for ascii = 97, 122 do
    --collectgarbage()
    --local c = string.char(ascii)
    --for i = 2, letter_cnt[c] do
      --local img = image.load(string.format('../alphabet_data/test/%s%d.jpg', c, i))
      --img = image.rgb2hsl(img)
      ----local ud_img = image.vflip(image.hflip(img))
      --for x = 4, 5 do
        --for y = 4, 5 do
          --if torch.rand(1)[1] < 0.8 then
            --for i = 0, 0 do
              --table.insert(train_dataset.data,
                           --image.scale(image.rotate(img[{{},{6*x+1,6*x+162},{6*y+1,6*y+162}}], i*90), geometry[1], geometry[2]):reshape(1, 3, geometry[1], geometry[2]))
              --table.insert(train_dataset.labels, ascii - 96)
            --end
          --else
            --for i = 0, 0 do
              --table.insert(dev_dataset.data,
                           --image.scale(image.rotate(img[{{},{6*x+1,6*x+162},{6*y+1,6*y+162}}], i*90), geometry[1], geometry[2]):reshape(1, 3, geometry[1], geometry[2]))
              --table.insert(dev_dataset.labels, ascii - 96)
            --end
          --end
        --end
      --end
    --end
  --end
  --train_dataset.data = nn.JoinTable(1):forward(train_dataset.data)
  ----torch.save('maomao_mean.th', torch.mean(train_dataset.data, 1))
  --print(torch.mean(train_dataset.data, 1))
  ----torch.save('maomao_std.th', torch.std(train_dataset.data, 1))
  --train_dataset.data = torch.cdiv(train_dataset.data - torch.mean(train_dataset.data, 1):expandAs(train_dataset.data), torch.std(train_dataset.data, 1):expandAs(train_dataset.data))
  --dev_dataset.data = nn.JoinTable(1):forward(dev_dataset.data)
  --dev_dataset.data = torch.cdiv(dev_dataset.data - torch.mean(dev_dataset.data, 1):expandAs(dev_dataset.data), torch.std(dev_dataset.data, 1):expandAs(dev_dataset.data))
  --train_dataset.labels = torch.IntTensor(train_dataset.labels):cuda()
  --dev_dataset.labels = torch.IntTensor(dev_dataset.labels):cuda()
  --train_dataset.size = train_dataset.labels:size(1)
  --dev_dataset.size = dev_dataset.labels:size(1)
  ----print(dev_dataset.data[1])
  ----print(dev_dataset.labels[1])
  --return train_dataset, dev_dataset
--end
-- test function
function test(dataset)
  model:evaluate()
  -- local vars
  local time = sys.clock()

  local acc = 0
  local avg_f = 0

  -- test over given dataset
  print('<trainer> on testing Set:')
  for t = 1,dataset.size,batchSize do

    -- create mini batch
    local batch_end = math.min(t+batchSize-1, dataset.size)
    local inputs = dataset.data[{{t,batch_end},{},{},{}}]:cuda()
    local targets = dataset.labels[{{t,batch_end}}]

    -- test samples
    local preds = model:forward(inputs)
    local f = criterion:forward(preds, targets)
    avg_f = avg_f + f

    --x, y = preds:topk(3, true, true)
    --print(x)
    --print(y)

    for i = 1, batch_end-t+1 do
      local _, idx = preds[i]:max(1)
      if idx[1] == targets[i] then
        acc = acc + 1
      end
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

  return acc, avg_f
end

----------------------------------------------------------------------
-- and train!
--
--
model = torch.load('maomao_hsv.th')

mean = torch.load('maomao_hsv_mean.th')
std = torch.load('maomao_hsv_std.th')
--train_dataset, dev_dataset = loadData()

--test(dev_dataset)
img = image.load('tmpimg/d2fed82325864648ab2bef65b42518c6.jpg') -- E
--img = image.load('tmpimg/7e0e217228e044ed85ff6541209b133f.jpg') -- G
--img = image.load('tmpimg/4074ec74d3c447ac825d5537f1500727.jpg') -- H
img = image.scale(img, 128, 128)
img = image.rgb2hsv(img)
img = torch.reshape(img, 1, 3, 128, 128)

img = torch.cdiv(img:csub(mean:expandAs(img)), std:expandAs(img)):cuda()
--print(img)

out = model:forward(img)
probs, indices = out:topk(10, true, true)
print(probs)
print(indices)

