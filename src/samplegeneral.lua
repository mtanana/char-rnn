
--[[

This file samples characters from a trained model

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

local WordSplitLMMinibatchLoader = require 'util.WordSplitLMMinibatchLoader'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',99,'random number generator\'s seed')
cmd:option('-formatwords',0,'Insert appropriate spaces in text')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',2000,'number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:text()


function translateWords(words)

  words=words:gsub("<c>","Client:\n");
  words=words:gsub("<t>","Therapist:\n");
  words=words:gsub("<eos>","\n");

  return words

end




local testphrases = {"<c> yeah, i'm having a really bad day <eos> <t> i'm sorry to hear that . how can I help ? <eos> ","<t> how are you today <eos> ","<t> what have you been up to ? <eos>"}

-- parse input params
opt = cmd:parse(arg)

-- gated print: simple utility function wrapping a print
function gprint(str)
  if opt.verbose == 1 then print(str) end
end

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 and opt.opencl == 0 then
  local ok, cunn = pcall(require, 'cunn')
  local ok2, cutorch = pcall(require, 'cutorch')
  if not ok then gprint('package cunn not found!') end
  if not ok2 then gprint('package cutorch not found!') end
  if ok and ok2 then
    gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
    gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    if opt.seed ~= 99 then cutorch.manualSeed(opt.seed) end
  else
    gprint('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

-- check that clnn/cltorch are installed if user wants to use OpenCL
if opt.gpuid >= 0 and opt.opencl == 1 then
  local ok, cunn = pcall(require, 'clnn')
  local ok2, cutorch = pcall(require, 'cltorch')
  if not ok then print('package clnn not found!') end
  if not ok2 then print('package cltorch not found!') end
  if ok and ok2 then
    gprint('using OpenCL on GPU ' .. opt.gpuid .. '...')
    gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
    cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    torch.manualSeed(opt.seed)
  else
    gprint('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
  gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state to all zeros
gprint('creating an ' .. checkpoint.opt.model .. '...')
local current_state
current_state = {}
for L = 1,checkpoint.opt.num_layers do
  -- c and h for all layers
  local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
  if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
  if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
  table.insert(current_state, h_init:clone())
  if checkpoint.opt.model == 'lstm' then
    table.insert(current_state, h_init:clone())
  end
end
state_size = #current_state

gprint("RNN Size: "..checkpoint.opt.rnn_size)
gprint("RNN N Layers: "..checkpoint.opt.num_layers)
gprint("Dropout: "..checkpoint.opt.dropout)

-- do a few seeded timesteps
local seed_text = opt.primetext
local unknownword = "<unk>"

function seed (seed_text)
  if string.len(seed_text) > 0 then
    --gprint('seeding with ' .. translateWords(seed_text))
    --gprint('--------------------------')


    local seedlist
    if(checkpoint.opt.wordlevel==1) then
      local words=WordSplitLMMinibatchLoader.preprocess(seed_text)
      seedlist = words:gmatch("([^%s]+)")
    else
      seedlist = seed_text:gmatch'.'

    end

    local seedtext = ""
    for c in seedlist do

      local idx = vocab[c]
      if idx == nil then
        idx = vocab[unknownword] 
        prev_char = torch.Tensor{vocab[unknownword]}
      else
       prev_char = torch.Tensor{vocab[c]}
      end
      
     
      seedtext=seedtext..ivocab[prev_char[1]]
      if(checkpoint.opt.wordlevel==1) then seedtext=seedtext.." " end
      if opt.gpuid >= 0 and opt.opencl == 0 then prev_char = prev_char:cuda() end
      if opt.gpuid >= 0 and opt.opencl == 1 then prev_char = prev_char:cl() end
      local lst = protos.rnn:forward{prev_char, unpack(current_state)}
      -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
      current_state = {}
      for i=1,state_size do table.insert(current_state, lst[i]) end
      prediction = lst[#lst] -- last element holds the log probabilities
    end
    --io.write(translateWords(seedtext))
  else
    -- fill with uniform probabilities over characters (? hmm)
    gprint('missing seed text, using uniform probability over first character')
    gprint('--------------------------')
    prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
    if opt.gpuid >= 0 and opt.opencl == 0 then prediction = prediction:cuda() end
    if opt.gpuid >= 0 and opt.opencl == 1 then prediction = prediction:cl() end
  end

end

--seed (seed_text)

function sample()
  -- start sampling/argmaxing
  local words=""
  for i=1, opt.length do

    -- log probabilities from the previous timestep
    if opt.sample == 0 then
      -- use argmax
      local _, prev_char_ = prediction:max(2)
      prev_char = prev_char_:resize(1)
    else
      -- use sampling
      prediction:div(opt.temperature) -- scale by temperature
      local probs = torch.exp(prediction):squeeze()
      probs:div(torch.sum(probs)) -- renormalize so probs sum to one
      prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
    end

    -- forward the rnn for next character
    local lst = protos.rnn:forward{prev_char, unpack(current_state)}
    current_state = {}
    for i=1,state_size do table.insert(current_state, lst[i]) end
    prediction = lst[#lst] -- last element holds the log probabilities
    local word = ivocab[prev_char[1]]
    if(checkpoint.opt.wordlevel==1) then
      if(word=="<eos>") then break end
      word=word.." "
    end
    words = words..word
  end
  words=translateWords(words)
  io.write(words)
  io.write('\n') io.flush()
end
--sample()
seed("<eos>")
--sample()
repeat
  io.write("Human: ")
  io.flush()
  answer=io.read()
  answer = " "..answer.." <eos> "
  io.write("Chat Bot: ")
  seed(answer)
  sample()
until answer=="end"