#!/usr/bin/env ruby
`rm -f perceptron.py`
%w{read_data.py preprocess_data.py train.py}.each do |file|
  `cat #{file} >> 'perceptron.py'`
end

`rm -f mlp.py`
%w{read_data.py preprocess_data.py mlp_train.py}.each do |file|
  `cat #{file} >> 'mlp.py'`
end

puts `python mlp.py`
