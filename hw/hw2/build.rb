#!/usr/bin/env ruby
`rm hw2.py`
%w{read_data.py preprocess_data.py train.py}.each do |file|
  `cat #{file} >> 'hw2.py'`
end


