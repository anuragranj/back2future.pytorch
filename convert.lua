-- Author: Anurag Ranjan
-- Copyright (c) 2018, Max Planck Society 

require 'paths'
require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'stn'
require 'spy'
require 'nngraph'
require 'models/CostVolMulti'

SAVE_DIR = 'pretrained/pwc'
model = torch.load('pretrained/Roaming_KITTI_model_300.t7' )

function save_sequential(name, model)
    for i = 1, #model do
        module = model:get(i)
        if tostring(torch.type(module)) == 'nn.SpatialConvolution' then
          torch.save(paths.concat(SAVE_DIR, name..'_'..tostring(i)..'weight.t7'), module.weight)
          torch.save(paths.concat(SAVE_DIR, name..'_'..tostring(i)..'bias.t7'), module.bias)
        end
    end
end

for i = 1, 200 do
    print('Traversing node' ..i)
    node = model:get(i)
    if tostring(torch.type(node)) == 'nn.Sequential' then
        nodenn = cudnn.convert(node, nn)
        nodenn_float = nodenn:float()
        save_sequential(tostring(i), nodenn_float)
    end
end

function warpingUnit()
  local I = nn.Identity()()
  local F = nn.Identity()()
  local input = I - nn.Transpose({2,3}, {3,4})
  local flow = F - nn.Transpose({2,3}, {3,4})
  local W = {input, flow} - nn.BilinearSamplerBHWD() - nn.Transpose({3,4}, {2,3})
  local model = nn.gModule({I, F}, {W})
  return model
end

for k, v in ipairs(model.forwardnodes) do
  print(k-1, v.id, v.data.module)
  v.data.annotations.name = tostring(k-1)
end
